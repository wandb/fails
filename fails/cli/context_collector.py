#!/usr/bin/env python3
"""
Interactive collector for user context about their AI system and evaluation.
Uses prompt_toolkit for dual text input areas.
"""

from typing import Optional, Tuple
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, VSplit, ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText


class UserContextCollector:
    """Collects user context about their AI system and evaluation."""
    
    def __init__(self, 
                 default_system_context: Optional[str] = None,
                 default_eval_context: Optional[str] = None):
        """Initialize the context collector.
        
        Args:
            default_system_context: Default text for AI system context
            default_eval_context: Default text for evaluation context
        """
        self.default_system_context = default_system_context
        self.default_eval_context = default_eval_context
        self.error_message = ""
        self.result = None
        self.current_focus = 0  # 0 for system context, 1 for eval context
        
    def run(self) -> Optional[Tuple[str, str]]:
        """Run the context collector and return the contexts.
        
        Returns:
            Tuple of (system_context, eval_context) or None if cancelled
        """
        # Create the two text input areas
        self.system_context_area = TextArea(
            text=self.default_system_context or "",
            multiline=True,
            wrap_lines=True,
            scrollbar=False,
            height=4,
            prompt="",
            focusable=True,
        )
        
        self.eval_context_area = TextArea(
            text=self.default_eval_context or "",
            multiline=True,
            wrap_lines=True,
            scrollbar=False,
            height=4,
            prompt="",
            focusable=True,
        )
        
        # Create header text
        header_text = FormattedText([
            ('class:step_header', '\nStep 2: Provide Context\n'),
            ('', '\n'),
            ('', 'Please provide context about your AI system and evaluation.\n'),
            ('class:instructions', 'This helps the pipeline better understand and categorize your failures.\n'),
            ('', '\n'),
            ('', '──────────────────────────────────────────────────────────────────────────────────────────────────────\n'),
            ('class:instructions', "Tab to switch fields | Enter to submit | Ctrl+C to cancel\n"),
            ('', '──────────────────────────────────────────────────────────────────────────────────────────────────────\n'),
            ('', '\n'),
        ])
        
        # Create frames for the text areas with titles
        system_frame = Frame(
            self.system_context_area,
            title="AI System Context - What task are you solving with AI?",
            width=105,
            height=6,
        )
        
        eval_frame = Frame(
            self.eval_context_area,
            title="Evaluation Context - What is this evaluation specifically testing?",
            width=105,
            height=6,
        )
        
        # Add example text below each frame
        system_example = FormattedText([
            ('class:instructions', '  Example: "Analyzing customer support transcripts to extract action items and sentiment"\n'),
        ])
        
        eval_example = FormattedText([
            ('class:instructions', '  Example: "Testing whether the model correctly identifies speaker roles (agent vs customer)"\n'),
        ])
        
        # Create error message window
        def get_error_text():
            return self.error_message if self.error_message else ""
        
        error_control = FormattedTextControl(
            text=get_error_text,
            focusable=False
        )
        
        # Left-align the frames
        left_aligned_system = VSplit([
            system_frame,
            Window(),  # Right padding
        ])
        
        left_aligned_eval = VSplit([
            eval_frame,
            Window(),  # Right padding
        ])
        
        # Create the layout
        root_container = HSplit([
            Window(FormattedTextControl(text=header_text), height=10),
            left_aligned_system,
            Window(FormattedTextControl(text=system_example), height=2),
            Window(height=1),  # Spacing
            left_aligned_eval,
            Window(FormattedTextControl(text=eval_example), height=2),
            ConditionalContainer(
                Window(error_control, height=3, style="class:error"),
                filter=Condition(lambda: bool(self.error_message))
            ),
        ])
        
        # Define key bindings
        kb = KeyBindings()
        
        @kb.add('c-c')
        def _(event):
            """Exit on Ctrl+C."""
            event.app.exit(result=None)
        
        @kb.add('tab')
        def _(event):
            """Switch between text areas on Tab."""
            if event.app.layout.current_window == self.system_context_area.window:
                event.app.layout.focus(self.eval_context_area)
            else:
                event.app.layout.focus(self.system_context_area)
        
        @kb.add('enter', filter=True)
        def _(event):
            """Submit on Enter if both contexts are provided."""
            system_text = self.system_context_area.text.strip()
            eval_text = self.eval_context_area.text.strip()
            
            if not system_text:
                self.error_message = "❌ Please provide AI system context."
                return
            
            if not eval_text:
                self.error_message = "❌ Please provide evaluation context."
                return
            
            # Both contexts provided, exit with result
            event.app.exit(result=(system_text, eval_text))
        
        # Define custom style
        style = Style.from_dict({
            'frame.border': '#888888',
            'error': 'fg:red bold',
            'instructions': 'fg:#888888',  # Grey color for instructions
            'step_header': 'fg:cyan bold',  # Cyan bold for step header
        })
        
        # Create and run the application
        app = Application(
            layout=Layout(root_container),
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=True,
        )
        
        # Set initial focus
        app.layout.focus(self.system_context_area)
        
        # Run the application and get the result
        result = app.run()
        
        # Print confirmation if we got valid results
        if result:
            system_context, eval_context = result
            print(f"\n\033[95m✓\033[0m Context collected successfully")
            print(f"  AI System: \033[96m{system_context[:50]}{'...' if len(system_context) > 50 else ''}\033[0m")
            print(f"  Evaluation: \033[96m{eval_context[:50]}{'...' if len(eval_context) > 50 else ''}\033[0m")
        
        return result


def collect_user_context(default_system: Optional[str] = None, 
                         default_eval: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Interactive function to collect user context.
    
    Args:
        default_system: Default AI system context
        default_eval: Default evaluation context
        
    Returns:
        Tuple of (system_context, eval_context) or None if cancelled
    """
    try:
        collector = UserContextCollector(
            default_system_context=default_system,
            default_eval_context=default_eval
        )
        return collector.run()
    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f"Error during context collection: {e}")
        return None


def main():
    """Test the context collector."""
    result = collect_user_context()
    
    if result:
        system_context, eval_context = result
        print("\nCollected contexts:")
        print(f"System: {system_context}")
        print(f"Eval: {eval_context}")
    else:
        print("Context collection cancelled.")


if __name__ == "__main__":
    main()