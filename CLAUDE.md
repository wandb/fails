# CLAUDE.md - Project Knowledge Base

## Terminal UI Arrow Selector Implementation

### Problem
Creating a working arrow-based selector for terminal UI that doesn't result in garbled/overlapping text. The initial attempts resulted in broken displays where text was overlapping and newlines weren't being handled correctly.

### Root Cause
When using raw terminal mode (`tty.setraw()`), the terminal doesn't automatically handle newlines (`\n`) properly. In raw mode:
- `\n` only moves down one line but doesn't return to the beginning of the line
- Terminal output can overlap if not properly cleared
- Standard print() functions don't work as expected

### Solution
The key insights that made it work:

1. **Use Alternate Screen Buffer**
   ```python
   sys.stdout.write("\033[?1049h")  # Enter alternate screen
   sys.stdout.write("\033[?1049l")  # Exit alternate screen
   ```
   This creates a separate screen (like vim/less do) that doesn't interfere with terminal scrollback.

2. **Clear Lines Before Writing**
   ```python
   sys.stdout.write("\033[K")  # Clear line from cursor to end
   ```
   This prevents text overlap by clearing each line before writing new content.

3. **Use Carriage Return + Line Feed in Raw Mode**
   ```python
   sys.stdout.write("text\r\n")  # NOT just \n
   ```
   In raw mode, we need explicit `\r\n` for proper line endings.

4. **Hide/Show Cursor**
   ```python
   sys.stdout.write("\033[?25l")  # Hide cursor
   sys.stdout.write("\033[?25h")  # Show cursor
   ```
   This prevents cursor flicker during redraws.

5. **Clear Remaining Screen**
   ```python
   sys.stdout.write("\033[J")  # Clear from cursor to end of screen
   ```
   This removes any leftover content below our UI.

### Working Implementation Pattern
```python
# Save terminal state
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)

try:
    # Setup
    sys.stdout.write("\033[?1049h\033[?25l")  # Alt screen + hide cursor
    tty.setraw(fd)
    
    while True:
        # Clear and redraw
        sys.stdout.write("\033[H")  # Move to home
        sys.stdout.write("\033[Kline 1\r\n")
        sys.stdout.write("\033[Kline 2\r\n")
        sys.stdout.write("\033[J")  # Clear rest
        sys.stdout.flush()
        
        # Handle input
        key = sys.stdin.read(1)
        if key == '\x1b':  # Handle arrow keys
            seq = sys.stdin.read(2)
            # Process escape sequences
            
finally:
    # Restore
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    sys.stdout.write("\033[?25h\033[?1049l")
    sys.stdout.flush()
```

### Key Escape Sequences Used
- `\033[H` - Move cursor to home (0,0)
- `\033[K` - Clear line from cursor to end
- `\033[J` - Clear screen from cursor to end
- `\033[?1049h` - Enter alternate screen buffer
- `\033[?1049l` - Exit alternate screen buffer
- `\033[?25l` - Hide cursor
- `\033[?25h` - Show cursor
- `\033[A` - Up arrow
- `\033[B` - Down arrow

### Lessons Learned
1. Don't use print() in raw mode - use sys.stdout.write()
2. Always use \r\n for line endings in raw mode
3. Clear lines before writing to prevent overlap
4. Use alternate screen buffer for full-screen TUIs
5. Always restore terminal settings in a finally block
6. Flush output after writes for immediate display

This approach creates a clean, flicker-free terminal UI that works across different terminal emulators.