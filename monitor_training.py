#!/usr/bin/env python3
# monitor_training.py - Real-time training monitoring with clean display

import os
import sys
import json
import time
import curses
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import subprocess

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    PYNVML_AVAILABLE = False

class TrainingMonitor:
    """Real-time training monitor with clean curses display"""
    
    def __init__(self, progress_file: str = "checkpoints/training_progress.json"):
        self.progress_file = progress_file
        self.last_modified = 0
        self.stats = {}
        self.start_time = time.time()
        
    def load_progress(self) -> bool:
        """Load progress from JSON file"""
        try:
            if not os.path.exists(self.progress_file):
                return False
                
            mtime = os.path.getmtime(self.progress_file)
            if mtime <= self.last_modified:
                return False
                
            with open(self.progress_file, 'r') as f:
                self.stats = json.load(f)
            self.last_modified = mtime
            return True
        except:
            return False
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        stats = {
            'util': 0,
            'memory_used': 0,
            'memory_total': 0,
            'temp': 0,
            'power': 0
        }
        
        if PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats['util'] = util.gpu
                
                # Memory
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats['memory_used'] = mem.used / 1e9
                stats['memory_total'] = mem.total / 1e9
                
                # Temperature
                stats['temp'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                stats['power'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                pass
                
        return stats
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to human readable time"""
        return str(timedelta(seconds=int(seconds)))
    
    def draw_interface(self, stdscr):
        """Draw the monitoring interface"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        
        # Colors
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Load latest progress
            self.load_progress()
            gpu_stats = self.get_gpu_stats()
            
            # Header
            header = "🚀 LLM TRAINING MONITOR"
            stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * width)
            
            row = 3
            
            # Training Progress
            if self.stats:
                tokens = self.stats.get('tokens_seen', 0)
                total_tokens = 3_000_000_000
                progress = (tokens / total_tokens) * 100
                current_day = (tokens // 500_000_000) + 1
                
                # Progress bar
                bar_width = width - 40
                filled = int(bar_width * progress / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                stdscr.addstr(row, 0, f"Progress: [{bar}] {progress:.1f}%", curses.color_pair(1))
                row += 1
                
                # Key metrics
                stdscr.addstr(row, 0, f"Day {current_day}/6 | Tokens: {tokens:,} / {total_tokens:,}")
                row += 2
                
                # Training metrics
                stdscr.addstr(row, 0, "TRAINING METRICS", curses.A_BOLD | curses.color_pair(4))
                row += 1
                stdscr.addstr(row, 0, "-" * 40)
                row += 1
                
                train_loss = self.stats.get('train_loss', 0)
                val_loss = self.stats.get('val_loss', 0)
                lr = self.stats.get('lr', 0)
                speed = self.stats.get('tokens_per_sec', 0)
                
                stdscr.addstr(row, 0, f"Train Loss: {train_loss:.4f}")
                if val_loss > 0:
                    stdscr.addstr(row, 25, f"Val Loss: {val_loss:.4f}")
                row += 1
                
                stdscr.addstr(row, 0, f"Learning Rate: {lr:.2e}")
                stdscr.addstr(row, 25, f"Speed: {speed:.0f} tok/s")
                row += 2
                
                # Time metrics
                elapsed = time.time() - self.start_time
                if speed > 0:
                    remaining_tokens = total_tokens - tokens
                    eta_seconds = remaining_tokens / speed
                    eta = self.format_time(eta_seconds)
                else:
                    eta = "N/A"
                
                stdscr.addstr(row, 0, f"Elapsed: {self.format_time(elapsed)}")
                stdscr.addstr(row, 25, f"ETA: {eta}")
                row += 2
            else:
                stdscr.addstr(row, 0, "Waiting for training data...", curses.color_pair(2))
                row += 2
            
            # GPU Stats
            stdscr.addstr(row, 0, "GPU STATUS", curses.A_BOLD | curses.color_pair(4))
            row += 1
            stdscr.addstr(row, 0, "-" * 40)
            row += 1
            
            # GPU utilization bar
            gpu_util = gpu_stats['util']
            gpu_bar_width = 20
            gpu_filled = int(gpu_bar_width * gpu_util / 100)
            gpu_bar = "█" * gpu_filled + "░" * (gpu_bar_width - gpu_filled)
            
            color = curses.color_pair(1) if gpu_util < 80 else curses.color_pair(2) if gpu_util < 95 else curses.color_pair(3)
            stdscr.addstr(row, 0, f"Utilization: [{gpu_bar}] {gpu_util}%", color)
            row += 1
            
            # Memory
            mem_used = gpu_stats['memory_used']
            mem_total = gpu_stats['memory_total']
            mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
            
            stdscr.addstr(row, 0, f"Memory: {mem_used:.1f}GB / {mem_total:.1f}GB ({mem_percent:.1f}%)")
            row += 1
            
            # Temperature and Power
            temp = gpu_stats['temp']
            power = gpu_stats['power']
            
            temp_color = curses.color_pair(1) if temp < 70 else curses.color_pair(2) if temp < 80 else curses.color_pair(3)
            stdscr.addstr(row, 0, f"Temperature: {temp}°C", temp_color)
            stdscr.addstr(row, 25, f"Power: {power:.0f}W")
            row += 2
            
            # Instructions
            stdscr.addstr(height - 2, 0, "Press 'q' to quit, 'r' to refresh", curses.A_DIM)
            
            # Refresh
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('r'):
                continue
                
            time.sleep(1)  # Update every second
    
    def run(self):
        """Run the monitor"""
        try:
            curses.wrapper(self.draw_interface)
        except KeyboardInterrupt:
            pass

def simple_monitor():
    """Simple text-based monitor for non-curses environments"""
    monitor = TrainingMonitor()
    
    print("Starting simple training monitor (Ctrl+C to stop)...")
    print("=" * 80)
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            monitor.load_progress()
            gpu_stats = monitor.get_gpu_stats()
            
            print("🚀 LLM TRAINING MONITOR")
            print("=" * 80)
            
            if monitor.stats:
                tokens = monitor.stats.get('tokens_seen', 0)
                total_tokens = 3_000_000_000
                progress = (tokens / total_tokens) * 100
                current_day = (tokens // 500_000_000) + 1
                
                print(f"Progress: {progress:.1f}% | Day {current_day}/6")
                print(f"Tokens: {tokens:,} / {total_tokens:,}")
                print()
                
                print("Training Metrics:")
                print(f"  Train Loss: {monitor.stats.get('train_loss', 0):.4f}")
                print(f"  Val Loss: {monitor.stats.get('val_loss', 0):.4f}")
                print(f"  Learning Rate: {monitor.stats.get('lr', 0):.2e}")
                print(f"  Speed: {monitor.stats.get('tokens_per_sec', 0):.0f} tok/s")
                print()
                
                print("GPU Status:")
                print(f"  Utilization: {gpu_stats['util']}%")
                print(f"  Memory: {gpu_stats['memory_used']:.1f}/{gpu_stats['memory_total']:.1f}GB")
                print(f"  Temperature: {gpu_stats['temp']}°C")
                print(f"  Power: {gpu_stats['power']:.0f}W")
            else:
                print("Waiting for training data...")
            
            print("=" * 80)
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    # Try curses interface first, fall back to simple
    try:
        monitor = TrainingMonitor()
        monitor.run()
    except:
        simple_monitor()