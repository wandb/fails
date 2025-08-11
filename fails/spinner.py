"""
Loading spinner for the FAILS pipeline.
"""

import sys
import time
import threading
import itertools
from typing import Optional


class FailsSpinner:
    """The Classic FAILS Rotator spinner for long-running operations."""
    
    def __init__(self, message: str = "Analyzing failures"):
        """Initialize the spinner.
        
        Args:
            message: The message to display alongside the spinner
        """
        self.frames = [
            'F A I L S',
            'A I L S F',
            'I L S F A',
            'L S F A I',
            'S F A I L',
        ]
        self.message = message
        self.spinner = itertools.cycle(self.frames)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
    def _spin(self):
        """Internal method to run the spinner animation."""
        # Add newlines before starting the spinner for breathing room
        if not hasattr(self, '_first_frame_shown'):
            sys.stdout.write('\n')
            # Print the first frame with extra lines below to push content up from bottom
            frame = next(self.spinner)
            colored_frame = self._colorize_frame(frame)
            sys.stdout.write(f'{colored_frame}  {self.message}...\n\n')
            sys.stdout.flush()
            self._first_frame_shown = True
            
        while not self._stop_event.is_set():
            frame = next(self.spinner)
            colored_frame = self._colorize_frame(frame)
            
            # Move cursor up 2 lines, clear line, print spinner, then move back down
            sys.stdout.write('\033[2A\r' + ' ' * 80 + '\r')  # Move up 2, clear line
            sys.stdout.write(f'{colored_frame}  {self.message}...\n\n')
            sys.stdout.flush()
            time.sleep(0.15)  # Adjust speed as needed
            
    def _colorize_frame(self, frame):
        """Helper method to colorize a spinner frame."""
        colored_frame = ""
        for i, char in enumerate(frame):
            if char == ' ':
                colored_frame += ' '
            elif i % 4 == 0:  # F and L positions
                colored_frame += f'\033[95m{char}\033[0m'  # bright_magenta
            elif i % 4 == 2:  # A and S positions  
                colored_frame += f'\033[96m{char}\033[0m'  # cyan
            else:  # I position
                colored_frame += char  # white
        return colored_frame
            
    def start(self):
        """Start the spinner in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        
    def stop(self, final_message: Optional[str] = None, success: bool = True):
        """Stop the spinner and optionally display a final message.
        
        Args:
            final_message: Optional message to display after stopping
            success: If True, shows a success indicator; if False, shows failure
        """
        if self._thread is None:
            return
            
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        
        # Move cursor up 2 lines to overwrite the spinner and clear the line
        sys.stdout.write('\033[2A\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        
        # Display final message if provided
        if final_message:
            if success:
                sys.stdout.write(f'\033[95m✓\033[0m {final_message}\n')
            else:
                sys.stdout.write(f'\033[91m✗\033[0m {final_message}\n')
            sys.stdout.flush()
        else:
            # If no final message, just leave cursor at cleared line
            sys.stdout.write('\n')
            sys.stdout.flush()
            
    def update_message(self, new_message: str):
        """Update the spinner message while it's running.
        
        Args:
            new_message: The new message to display
        """
        self.message = new_message
        
    def __enter__(self):
        """Context manager entry - starts the spinner."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the spinner."""
        # Determine success based on whether an exception occurred
        success = exc_type is None
        self.stop(success=success)
        

# Convenience function for simple usage
def with_spinner(message: str = "Processing"):
    """Decorator to show a spinner while a function executes.
    
    Usage:
        @with_spinner("Loading data")
        def my_function():
            time.sleep(3)
            return "done"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            spinner = FailsSpinner(message)
            spinner.start()
            try:
                result = func(*args, **kwargs)
                spinner.stop(f"{message} complete", success=True)
                return result
            except Exception:
                spinner.stop(f"{message} failed", success=False)
                raise
        return wrapper
    return decorator


# Example usage functions
def demo():
    """Demonstrate the spinner usage."""
    print("Demo 1: Basic usage")
    spinner = FailsSpinner("Analyzing evaluation traces")
    spinner.start()
    time.sleep(3)
    spinner.stop("Analysis complete", success=True)
    
    print("\nDemo 2: Context manager")
    with FailsSpinner("Categorizing failures"):
        time.sleep(3)
    print("Done!")
    
    print("\nDemo 3: With message updates")
    spinner = FailsSpinner("Initializing")
    spinner.start()
    time.sleep(1)
    spinner.update_message("Detecting failures")
    time.sleep(1)
    spinner.update_message("Generating report")
    time.sleep(1)
    spinner.stop("Pipeline complete", success=True)
    
    print("\nDemo 4: Decorator usage")
    @with_spinner("Processing data")
    def slow_function():
        time.sleep(2)
        return "Result"
    
    result = slow_function()
    print(f"Got result: {result}")


if __name__ == "__main__":
    demo()