#!/usr/bin/env python3
"""
Shared FAILS header for consistent display across all CLI screens.
"""

# ANSI color codes
BRIGHT_MAGENTA = "\033[1m\033[95m"
BRIGHT_CYAN = "\033[1m\033[96m"
RESET = "\033[0m"

# The FAILS header as a string (for raw terminal output)
FAILS_HEADER_RAW = f"""{BRIGHT_MAGENTA}   ┌───────────────────────────────────────────────────────────────────────────┐{RESET}
{BRIGHT_MAGENTA}   │{RESET}                                                                           {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                                                                           {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                      {BRIGHT_CYAN}███████╗ █████╗ ██╗██╗     ███████╗{RESET}                  {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                      {BRIGHT_CYAN}██╔════╝██╔══██╗██║██║     ██╔════╝{RESET}                  {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                      {BRIGHT_CYAN}█████╗  ███████║██║██║     ███████╗{RESET}                  {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                      {BRIGHT_CYAN}██╔══╝  ██╔══██║██║██║     ╚════██║{RESET}                  {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                      {BRIGHT_CYAN}██║     ██║  ██║██║███████╗███████║{RESET}                  {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                      {BRIGHT_CYAN}╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝{RESET}                  {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                                                                           {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   │{RESET}                                                                           {BRIGHT_MAGENTA}│{RESET}
{BRIGHT_MAGENTA}   └───────────────────────────────────────────────────────────────────────────┘{RESET}"""


def print_fails_header():
    """Print the FAILS header to stdout."""
    print(FAILS_HEADER_RAW)


def get_fails_header_lines():
    """Get the FAILS header as a list of lines (without newlines)."""
    return FAILS_HEADER_RAW.split('\n')


def get_fails_header_for_raw_terminal():
    """Get the FAILS header formatted for raw terminal mode with \r\n line endings."""
    lines = []
    for line in get_fails_header_lines():
        lines.append(line + "\r\n")
    return ''.join(lines)


def get_fails_header_for_prompt_toolkit():
    """Get the FAILS header as a list of tuples for prompt_toolkit FormattedText."""
    return [
        ('class:logo_border', '   ┌───────────────────────────────────────────────────────────────────────────┐\n'),
        ('class:logo_border', '   │'),
        ('', '                                                                           '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                                                                           '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                      '),
        ('class:logo_text', '███████╗ █████╗ ██╗██╗     ███████╗'),
        ('', '                  '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                      '),
        ('class:logo_text', '██╔════╝██╔══██╗██║██║     ██╔════╝'),
        ('', '                  '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                      '),
        ('class:logo_text', '█████╗  ███████║██║██║     ███████╗'),
        ('', '                  '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                      '),
        ('class:logo_text', '██╔══╝  ██╔══██║██║██║     ╚════██║'),
        ('', '                  '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                      '),
        ('class:logo_text', '██║     ██║  ██║██║███████╗███████║'),
        ('', '                  '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                      '),
        ('class:logo_text', '╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝'),
        ('', '                  '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                                                                           '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   │'),
        ('', '                                                                           '),
        ('class:logo_border', '│\n'),
        ('class:logo_border', '   └───────────────────────────────────────────────────────────────────────────┘\n'),
    ]


def get_fails_header_for_rich():
    """Get the FAILS header formatted for Rich console output."""
    return """[bold bright_magenta]   ┌───────────────────────────────────────────────────────────────────────────┐[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                                                                           [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                                                                           [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                      [bold cyan]███████╗ █████╗ ██╗██╗     ███████╗[/bold cyan]                  [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                      [bold cyan]██╔════╝██╔══██╗██║██║     ██╔════╝[/bold cyan]                  [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                      [bold cyan]█████╗  ███████║██║██║     ███████╗[/bold cyan]                  [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                      [bold cyan]██╔══╝  ██╔══██║██║██║     ╚════██║[/bold cyan]                  [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                      [bold cyan]██║     ██║  ██║██║███████╗███████║[/bold cyan]                  [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                      [bold cyan]╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝[/bold cyan]                  [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                                                                           [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   │[/bold bright_magenta]                                                                           [bold bright_magenta]│[/bold bright_magenta]
[bold bright_magenta]   └───────────────────────────────────────────────────────────────────────────┘[/bold bright_magenta]"""