import numpy as np
import sounddevice as sd
from pynput import keyboard
import time
import threading
import pygame

# --- Configuration ---
SAMPLE_RATE = 44100
TARGET_FPS = 60
POINTS_PER_FRAME = int(SAMPLE_RATE / TARGET_FPS)

# Game Constants
PADDLE_HEIGHT = 0.2
PADDLE_SPEED = 0.04
BALL_SIZE = 0.03
GRAVITY = 0.0008
BOUNCE_ELASTICITY = 0.85

# Viewer Constants
WINDOW_SIZE = 800
BEAM_COLOR = (0, 255, 64)  # Oscilloscope Green
BG_COLOR = (10, 20, 10)    # Dark Greenish Black

# --- Game State ---
class GameState:
    def __init__(self):
        self.reset_game()
        self.running = True
        self.keys_pressed = set()
        self.current_buffer = np.zeros((POINTS_PER_FRAME, 2), dtype=np.float32)
        self.buffer_lock = threading.Lock()

    def reset_game(self):
        self.reset_ball()
        self.p1_y = 0.0
        self.p2_y = 0.0
        self.score1 = 0
        self.score2 = 0
        print("Game Restarted!")

    def reset_ball(self):
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.015 * (1 if np.random.rand() > 0.5 else -1)
        self.ball_vy = 0.01

    def update_logic(self):
        # Paddle movement
        if 'w' in self.keys_pressed:
            self.p1_y = min(1.0 - PADDLE_HEIGHT, self.p1_y + PADDLE_SPEED)
        if 's' in self.keys_pressed:
            self.p1_y = max(-0.9 + PADDLE_HEIGHT, self.p1_y - PADDLE_SPEED)
        
        if keyboard.Key.up in self.keys_pressed:
            self.p2_y = min(1.0 - PADDLE_HEIGHT, self.p2_y + PADDLE_SPEED)
        if keyboard.Key.down in self.keys_pressed:
            self.p2_y = max(-0.9 + PADDLE_HEIGHT, self.p2_y - PADDLE_SPEED)

        # Ball physics
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        self.ball_vy -= GRAVITY

        if self.ball_y > 1.0:
            self.ball_y = 1.0
            self.ball_vy *= -BOUNCE_ELASTICITY
        if self.ball_y < -0.9:
            self.ball_y = -0.9
            self.ball_vy *= -BOUNCE_ELASTICITY
            
        if abs(self.ball_x) < 0.03 and self.ball_y < -0.6:
            self.ball_vx *= -0.7

        if self.ball_x < -0.9 and abs(self.ball_y - self.p1_y) < PADDLE_HEIGHT:
            self.ball_x = -0.9
            self.ball_vx = abs(self.ball_vx) * 1.05
            self.ball_vy += (self.ball_y - self.p1_y) * 0.05
        if self.ball_x > 0.9 and abs(self.ball_y - self.p2_y) < PADDLE_HEIGHT:
            self.ball_x = 0.9
            self.ball_vx = -abs(self.ball_vx) * 1.05
            self.ball_vy += (self.ball_y - self.p2_y) * 0.05

        if self.ball_x < -1.1:
            self.score2 += 1
            print(f"P1: {self.score1} | P2: {self.score2}")
            self.reset_ball()
        elif self.ball_x > 1.1:
            self.score1 += 1
            print(f"P1: {self.score1} | P2: {self.score2}")
            self.reset_ball()

    def generate_frame(self):
        paths = []
        
        # Draw objects multiple times or with more points to make them brighter
        # 1. Court (Bottom line)
        paths.append(get_line_points((-1, -0.9), (1, -0.9), 150))
        
        # 2. Net
        paths.append(get_line_points((0, -0.9), (0, -0.6), 60))
        
        # 3. Paddle 1
        paths.append(get_line_points((-0.95, self.p1_y - PADDLE_HEIGHT), 
                                     (-0.95, self.p1_y + PADDLE_HEIGHT), 100))
        
        # 4. Paddle 2
        paths.append(get_line_points((0.95, self.p2_y - PADDLE_HEIGHT), 
                                     (0.95, self.p2_y + PADDLE_HEIGHT), 100))
        
        # 5. Ball
        bs = BALL_SIZE
        ball_path = [
            (self.ball_x - bs, self.ball_y),
            (self.ball_x, self.ball_y + bs),
            (self.ball_x + bs, self.ball_y),
            (self.ball_x, self.ball_y - bs),
            (self.ball_x - bs, self.ball_y)
        ]
        ball_pts = []
        for i in range(len(ball_path)-1):
            ball_pts.append(get_line_points(ball_path[i], ball_path[i+1], 25))
        paths.append(np.vstack(ball_pts))

        # Combine with MINIMAL blanking points
        # Fewer points = faster beam = dimmer line
        full_buffer = []
        for i, path in enumerate(paths):
            full_buffer.append(path)
            if i < len(paths) - 1:
                full_buffer.append(get_line_points(path[-1], paths[i+1][0], 5))
        
        full_buffer.append(get_line_points(paths[-1][-1], paths[0][0], 5))
        combined = np.vstack(full_buffer)
        
        current_len = len(combined)
        indices = np.linspace(0, current_len - 1, POINTS_PER_FRAME)
        resampled = np.zeros((POINTS_PER_FRAME, 2), dtype=np.float32)
        resampled[:, 0] = np.interp(indices, np.arange(current_len), combined[:, 0])
        resampled[:, 1] = np.interp(indices, np.arange(current_len), combined[:, 1])
        
        with self.buffer_lock:
            self.current_buffer = resampled

def get_line_points(p1, p2, num_points):
    return np.array([
        np.linspace(p1[0], p2[0], num_points),
        np.linspace(p1[1], p2[1], num_points)
    ]).T

game = GameState()

# --- Input Handling ---

def on_press(key):
    try:
        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            game.keys_pressed.add(k)
            if k == 'r':
                game.reset_game()
        else:
            game.keys_pressed.add(key)
    except AttributeError:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char:
            game.keys_pressed.discard(key.char)
        else:
            game.keys_pressed.discard(key)
    except AttributeError:
        pass
    if key == keyboard.Key.esc:
        game.running = False

# --- Audio Callback ---

playback_ptr = 0

def audio_callback(outdata, frames, time_info, status):
    global playback_ptr
    if status:
        print(status)
    with game.buffer_lock:
        buf = game.current_buffer
        buf_len = len(buf)
        indices = (np.arange(playback_ptr, playback_ptr + frames)) % buf_len
        outdata[:] = buf[indices]
        playback_ptr = (playback_ptr + frames) % buf_len

# --- Main ---

print("Starting Tennis for Two (Oscilloscope XY Mode + Simulator)...")
print("Controls: P1 (W/S) | P2 (Up/Down Arrows)")
print("Restart: R | Quit: ESC")

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Start game loop
def game_loop():
    while game.running:
        start_time = time.time()
        game.update_logic()
        game.generate_frame()
        elapsed = time.time() - start_time
        time.sleep(max(0, (1.0 / TARGET_FPS) - elapsed))

logic_thread = threading.Thread(target=game_loop)
logic_thread.daemon = True
logic_thread.start()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Tennis for Two - Oscilloscope Simulator")
clock = pygame.time.Clock()

# Create a persistence surface for the "phosphor glow"
persistence_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
persistence_surface.set_alpha(20) # Controls how long the "trail" lasts

def world_to_screen(point):
    x = int((point[0] + 1) / 2 * WINDOW_SIZE)
    y = int((1 - point[1]) / 2 * WINDOW_SIZE)
    return (x, y)

try:
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLE_RATE):
        while game.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.running = False

            # 1. Fade out the screen (persistence)
            # Instead of filling with BG_COLOR, we fill with a slightly transparent black
            # This creates a "glow" effect as old lines stay for a few frames
            black_overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            black_overlay.fill((0, 5, 0)) # Very dark green fade
            black_overlay.set_alpha(40)   # Higher = shorter persistence
            screen.blit(black_overlay, (0,0))
            
            with game.buffer_lock:
                buf = game.current_buffer.copy()
            
            if len(buf) > 1:
                # To simulate the beam, we draw every point segment
                last_p = world_to_screen(buf[0])
                for i in range(1, len(buf)):
                    p = world_to_screen(buf[i])
                    
                    # Calculate beam velocity (distance between points)
                    # Normalized distance for better blanking detection
                    dx = buf[i][0] - buf[i-1][0]
                    dy = buf[i][1] - buf[i-1][1]
                    vel = np.sqrt(dx*dx + dy*dy)
                    
                    # Real Scope: fast movement = less light
                    if vel < 0.05:
                        brightness = 255
                        width = 2
                    elif vel < 0.2:
                        brightness = 100
                        width = 1
                    else:
                        brightness = 20 # Very faint "blanking" line
                        width = 1
                    
                    color = (0, brightness, 0)
                    pygame.draw.line(screen, color, last_p, p, width)
                    
                    # Add a "core" bright line for game objects
                    if vel < 0.05:
                        pygame.draw.line(screen, (150, 255, 150), last_p, p, 1)
                        
                    last_p = p

            pygame.display.flip()
            clock.tick(TARGET_FPS)
except Exception as e:
    print(f"Error: {e}")
finally:
    game.running = False
    pygame.quit()
    listener.stop()
    logic_thread.join()
    print("Game exited.")

