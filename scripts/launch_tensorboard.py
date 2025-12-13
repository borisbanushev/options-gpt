#!/usr/bin/env python3
"""
Launch TensorBoard to monitor training progress.
"""

import os
import subprocess
import sys
from pathlib import Path

def launch_tensorboard(log_dir: str = None):
    """Launch TensorBoard with the specified log directory."""
    if log_dir is None:
        log_dir = "./artifacts/fine_tuned/tensorboard_logs"
    
    # Check if log directory exists
    if not Path(log_dir).exists():
        print(f"❌ TensorBoard log directory not found: {log_dir}")
        print("💡 Make sure training has started or specify the correct path")
        return
    
    print(f"🚀 Launching TensorBoard for log directory: {log_dir}")
    print("📊 Open your browser to: http://localhost:6006")
    print("⏹️  Press Ctrl+C to stop TensorBoard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--port", "6006",
            "--host", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard stopped")
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch TensorBoard for training monitoring")
    parser.add_argument("--logdir", default=None, help="TensorBoard log directory")
    args = parser.parse_args()
    
    launch_tensorboard(args.logdir)
