#!/usr/bin/env bash
set -e

pip install -q numpy sounddevice pynput pygame
python3 "$(dirname "$0")/tennis_for_two.py"
