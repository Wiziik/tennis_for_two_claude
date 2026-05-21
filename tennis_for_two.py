#!/usr/bin/env python3
"""
Tennis for Two — oscilloscope XY renderer
==========================================
Connect stereo audio out to oscilloscope set to X-Y mode.
  Left  channel → X axis
  Right channel → Y axis

Install:  pip install numpy sounddevice pynput pygame
Run:      python3 tennis_for_two.py
Controls: W / S = left paddle   ↑ / ↓ = right paddle   ESC = quit
"""

import sys
import time
import queue
import threading

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sys.exit("Missing: pip install sounddevice")

try:
    from pynput import keyboard as kb
except ImportError:
    sys.exit("Missing: pip install pynput")

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False
    print("pygame not found — running without viewer. (pip install pygame)")


# ── Tunable constants ─────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100
BALL_SPEED    = 0.014     # units/frame  (frame rate ≈ 92 fps)
PADDLE_SPEED  = 0.030     # units/frame
PADDLE_HALF   = 0.16      # half-height of each paddle
PADDLE_X      = 0.82      # x-position of both paddles (mirrored)
COURT_BOTTOM  = -0.80
COURT_TOP     =  0.85
NET_TOP       = -0.35     # top of visual net (not a collision surface)

# Samples per rendered element — more = brighter/smoother, fewer = faster refresh
COURT_N  = 160
NET_N    =  55
BALL_N   =  20   # circle subdivisions (+1 added to close the loop)
PADDLE_N =  75
RAMP_N   =  18   # blanking transition between shapes

# Viewer
VIEW_W, VIEW_H  = 700, 700
PHOSPHOR_DECAY  = 0.82    # per-frame brightness multiplier (0=instant off, 1=never fades)
PHOSPHOR_PEAK   = 2.8     # accumulated brightness that maps to full white-green
GLOW_INTENSITY  = 0.30    # brightness added to 4-neighbour pixels


# ── Shared game state ─────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.bx, self.by = 0.0, 0.0
        self.vx = BALL_SPEED
        self.vy = BALL_SPEED * 0.55
        self.ly = 0.0
        self.ry = 0.0
        self.sl = 0
        self.sr = 0
        self.keys = {'w': False, 's': False, 'up': False, 'down': False}
        self.key_lock = threading.Lock()

S = State()

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
_leftover = np.zeros((0, 2), dtype=np.float32)


# ── Vector drawing helpers ────────────────────────────────────────────────────

def seg(x0: float, y0: float, x1: float, y1: float, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(n, 2), dtype=np.float32)
    return np.column_stack([x0 + (x1 - x0) * t,
                             y0 + (y1 - y0) * t])


def circle(cx: float, cy: float, r: float, n: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    pts = np.column_stack([cx + r * np.cos(t),
                           cy + r * np.sin(t)])
    return np.vstack([pts, pts[:1]])


def build_frame(bx: float, by: float, ly: float, ry: float) -> np.ndarray:
    p = []

    court = seg(-0.90, COURT_BOTTOM, 0.90, COURT_BOTTOM, COURT_N)
    p.append(court)

    p.append(seg(*court[-1], 0.0, COURT_BOTTOM, RAMP_N))
    net = seg(0.0, COURT_BOTTOM, 0.0, NET_TOP, NET_N)
    p.append(net)

    p.append(seg(*net[-1], bx, by, RAMP_N))
    ball = circle(bx, by, 0.035, BALL_N)
    p.append(ball)

    p.append(seg(*ball[-1], -PADDLE_X, ly + PADDLE_HALF, RAMP_N))
    lpad = seg(-PADDLE_X, ly + PADDLE_HALF, -PADDLE_X, ly - PADDLE_HALF, PADDLE_N)
    p.append(lpad)

    p.append(seg(*lpad[-1], PADDLE_X, ry + PADDLE_HALF, RAMP_N))
    rpad = seg(PADDLE_X, ry + PADDLE_HALF, PADDLE_X, ry - PADDLE_HALF, PADDLE_N)
    p.append(rpad)

    p.append(seg(*rpad[-1], -0.90, COURT_BOTTOM, RAMP_N))

    return np.clip(np.concatenate(p), -1.0, 1.0).astype(np.float32)


# ── Audio callback ────────────────────────────────────────────────────────────

def audio_cb(outdata: np.ndarray, frames: int, time_info, status) -> None:
    global _leftover
    buf = _leftover
    while len(buf) < frames:
        try:
            buf = np.concatenate([buf, audio_q.get_nowait()])
        except queue.Empty:
            pad = np.empty((frames - len(buf), 2), dtype=np.float32)
            pad[:] = buf[-1] if len(buf) else 0.0
            buf = np.concatenate([buf, pad])
            break
    outdata[:] = buf[:frames]
    _leftover  = buf[frames:]


# ── Keyboard ──────────────────────────────────────────────────────────────────

def _key(name: str, val: bool):
    with S.key_lock:
        S.keys[name] = val


def on_press(key):
    if   key == kb.KeyCode.from_char('w'): _key('w',    True)
    elif key == kb.KeyCode.from_char('s'): _key('s',    True)
    elif key == kb.Key.up:                  _key('up',   True)
    elif key == kb.Key.down:                _key('down', True)
    elif key == kb.KeyCode.from_char('r'): _restart()
    elif key == kb.Key.esc:                 return False


def on_release(key):
    if   key == kb.KeyCode.from_char('w'): _key('w',    False)
    elif key == kb.KeyCode.from_char('s'): _key('s',    False)
    elif key == kb.Key.up:                  _key('up',   False)
    elif key == kb.Key.down:                _key('down', False)


# ── Game logic ────────────────────────────────────────────────────────────────

def _restart():
    S.sl, S.sr = 0, 0
    S.ly, S.ry = 0.0, 0.0
    _reset_ball()
    _print_score()


def _reset_ball():
    S.bx, S.by = 0.0, 0.0
    S.vx = BALL_SPEED * (1 if np.random.rand() > 0.5 else -1)
    S.vy = BALL_SPEED * 0.55 * (1 if np.random.rand() > 0.5 else -1)


def tick() -> None:
    with S.key_lock:
        keys = dict(S.keys)

    if keys['w']:    S.ly = min(S.ly + PADDLE_SPEED, COURT_TOP    - PADDLE_HALF)
    if keys['s']:    S.ly = max(S.ly - PADDLE_SPEED, COURT_BOTTOM + PADDLE_HALF)
    if keys['up']:   S.ry = min(S.ry + PADDLE_SPEED, COURT_TOP    - PADDLE_HALF)
    if keys['down']: S.ry = max(S.ry - PADDLE_SPEED, COURT_BOTTOM + PADDLE_HALF)

    S.bx += S.vx
    S.by += S.vy

    if S.by >= COURT_TOP:
        S.by = COURT_TOP;    S.vy = -abs(S.vy)
    if S.by <= COURT_BOTTOM:
        S.by = COURT_BOTTOM; S.vy =  abs(S.vy)

    if S.vx < 0 and S.bx <= -PADDLE_X + 0.06:
        if abs(S.by - S.ly) <= PADDLE_HALF:
            S.vx  = abs(S.vx)
            S.vy += (S.by - S.ly) / PADDLE_HALF * BALL_SPEED * 0.7

    if S.vx > 0 and S.bx >= PADDLE_X - 0.06:
        if abs(S.by - S.ry) <= PADDLE_HALF:
            S.vx  = -abs(S.vx)
            S.vy += (S.by - S.ry) / PADDLE_HALF * BALL_SPEED * 0.7

    if S.bx < -1.1:
        S.sr += 1
        _print_score()
        _reset_ball()
    elif S.bx > 1.1:
        S.sl += 1
        _print_score()
        _reset_ball()


def _print_score():
    print(f"\r  Left {S.sl} : {S.sr} Right   ", end='', flush=True)


# ── Phosphor viewer ───────────────────────────────────────────────────────────

# phosphor buffer is (W, H) float32 — x-first to match pygame.surfarray layout
_phosphor = np.zeros((VIEW_W, VIEW_H), dtype=np.float32)
_screen   = None
_surf     = None
_font     = None


def viewer_init() -> None:
    global _screen, _surf, _font
    pygame.init()
    _screen = pygame.display.set_mode((VIEW_W, VIEW_H))
    pygame.display.set_caption("Tennis for Two — XY Oscilloscope")
    _surf  = pygame.Surface((VIEW_W, VIEW_H))
    _font  = pygame.font.SysFont("monospace", 22, bold=True)


def viewer_draw(frame: np.ndarray) -> bool:
    """
    Render one XY frame onto the phosphor display.
    Returns False when the window is closed so the game loop can exit.
    """
    global _phosphor

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False

    # ── Phosphor decay ────────────────────────────────────────────────────────
    _phosphor *= PHOSPHOR_DECAY

    # ── Map XY samples → pixel coordinates ───────────────────────────────────
    # X: -1..+1  →  0..VIEW_W-1   (left to right)
    # Y: -1..+1  →  VIEW_H-1..0   (bottom to top, flip y)
    xs = np.clip(
        ((frame[:, 0] + 1.0) * 0.5 * (VIEW_W - 1)).astype(np.int32),
        0, VIEW_W - 1,
    )
    ys = np.clip(
        ((1.0 - (frame[:, 1] + 1.0) * 0.5) * (VIEW_H - 1)).astype(np.int32),
        0, VIEW_H - 1,
    )

    # np.add.at handles duplicate indices correctly (unlike plain +=)
    np.add.at(_phosphor, (xs, ys), 1.0)

    # 4-neighbour glow
    np.add.at(_phosphor, (np.clip(xs + 1, 0, VIEW_W - 1), ys), GLOW_INTENSITY)
    np.add.at(_phosphor, (np.clip(xs - 1, 0, VIEW_W - 1), ys), GLOW_INTENSITY)
    np.add.at(_phosphor, (xs, np.clip(ys + 1, 0, VIEW_H - 1)), GLOW_INTENSITY)
    np.add.at(_phosphor, (xs, np.clip(ys - 1, 0, VIEW_H - 1)), GLOW_INTENSITY)

    # ── Phosphor → RGB ────────────────────────────────────────────────────────
    b = np.clip(_phosphor / PHOSPHOR_PEAK, 0.0, 1.0)  # 0..1 normalised

    rgb = np.zeros((VIEW_W, VIEW_H, 3), dtype=np.uint8)
    rgb[:, :, 1] = (b * 230).astype(np.uint8)          # green  (dominant)
    rgb[:, :, 0] = (b * b * 55).astype(np.uint8)       # red    (warm secondary glow)
    rgb[:, :, 2] = (b * b * 25).astype(np.uint8)       # blue   (subtle)

    pygame.surfarray.blit_array(_surf, rgb)             # surfarray wants (W, H, 3)
    _screen.blit(_surf, (0, 0))

    # ── Score overlay ─────────────────────────────────────────────────────────
    score_text = f"{S.sl}   :   {S.sr}"
    label = _font.render(score_text, True, (0, 200, 60))
    _screen.blit(label, (VIEW_W // 2 - label.get_width() // 2, 10))

    ctrl = _font.render("W/S  ·  ↑/↓  ·  R=restart  ·  ESC", True, (0, 80, 30))
    _screen.blit(ctrl, (VIEW_W // 2 - ctrl.get_width() // 2, VIEW_H - 30))

    pygame.display.flip()
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(__doc__)
    _print_score()

    if PYGAME_OK:
        viewer_init()

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype='float32',
        blocksize=256,
        callback=audio_cb,
        latency='low',
    )
    stream.start()

    _probe   = build_frame(0.0, 0.0, 0.0, 0.0)
    frame_dt = len(_probe) / SAMPLE_RATE

    for _ in range(4):
        audio_q.put(build_frame(S.bx, S.by, S.ly, S.ry))

    running = True
    try:
        while listener.is_alive() and running:
            t0 = time.perf_counter()

            tick()
            frame = build_frame(S.bx, S.by, S.ly, S.ry)

            try:
                audio_q.put(frame, timeout=0.05)
            except queue.Full:
                pass

            if PYGAME_OK:
                running = viewer_draw(frame)

            elapsed = time.perf_counter() - t0
            wait = frame_dt - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        listener.stop()
        if PYGAME_OK:
            pygame.quit()
        print("\n\nGame over.")


if __name__ == '__main__':
    main()
