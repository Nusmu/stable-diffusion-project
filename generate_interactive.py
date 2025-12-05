#!/usr/bin/env python3
"""
Interactive Stable Diffusion CLI

Conversational image generation with refinement and history.
"""

import os
import sys
import requests
from pathlib import Path
from datetime import datetime


class ImageGenerator:
    def __init__(self, token: str, output_path: str = "/app/output/current.png"):
        self.token = token
        self.output_path = Path(output_path)
        self.model = "black-forest-labs/FLUX.1-schnell"
        self.history = []
        self.current_prompt = ""

    def generate(self, prompt: str) -> bool:
        """Generate an image from prompt."""
        API_URL = f"https://router.huggingface.co/hf-inference/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": prompt}

        print(f"\n🎨 Generating with: {self.model}")
        print(f"📝 Prompt: {prompt}\n")

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if response.status_code != 200:
                error = response.json() if 'json' in response.headers.get('content-type', '') else response.text
                print(f"❌ API error {response.status_code}: {error}")
                return False

            self.output_path.write_bytes(response.content)
            self.current_prompt = prompt
            self.history.append(prompt)
            print(f"✅ Saved to: {self.output_path}")
            return True

        except requests.exceptions.Timeout:
            print("❌ Request timed out. Try again.")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def save_copy(self, name: str = None):
        """Save a copy of current image."""
        if not self.output_path.exists():
            print("❌ No image to save")
            return

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"image_{timestamp}.png"

        if not name.endswith('.png'):
            name += '.png'

        dest = self.output_path.parent / name
        dest.write_bytes(self.output_path.read_bytes())
        print(f"💾 Saved copy to: {dest}")

    def show_help(self):
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                    INTERACTIVE IMAGE GENERATOR                   ║
╠══════════════════════════════════════════════════════════════════╣
║  GENERATING IMAGES:                                              ║
║    Just type your prompt and press Enter                         ║
║                                                                  ║
║  REFINING:                                                       ║
║    + add more detail     → Adds to current prompt                ║
║    - remove something    → Removes from current prompt           ║
║    ! new complete prompt → Replaces entire prompt                ║
║                                                                  ║
║  COMMANDS:                                                       ║
║    /save [name]   Save a copy of current image                   ║
║    /prompt        Show current prompt                            ║
║    /history       Show prompt history                            ║
║    /model [name]  Switch model (schnell/dev)                     ║
║    /help          Show this help                                 ║
║    /quit          Exit                                           ║
║                                                                  ║
║  EXAMPLES:                                                       ║
║    > a red sports car on a mountain road                         ║
║    > + with sunset lighting                                      ║
║    > + photorealistic, 8k                                        ║
║    > - mountain                                                  ║
║    > ! a blue motorcycle in the city                             ║
╚══════════════════════════════════════════════════════════════════╝
""")

    def process_input(self, user_input: str) -> bool:
        """Process user input. Returns False to quit."""
        text = user_input.strip()

        if not text:
            return True

        # Commands
        if text.startswith('/'):
            parts = text[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ('quit', 'exit', 'q'):
                print("👋 Goodbye!")
                return False
            elif cmd == 'help':
                self.show_help()
            elif cmd == 'prompt':
                if self.current_prompt:
                    print(f"📝 Current prompt: {self.current_prompt}")
                else:
                    print("📝 No prompt yet")
            elif cmd == 'history':
                if self.history:
                    print("📜 Prompt history:")
                    for i, p in enumerate(self.history, 1):
                        print(f"  {i}. {p}")
                else:
                    print("📜 No history yet")
            elif cmd == 'save':
                self.save_copy(arg)
            elif cmd == 'model':
                if arg:
                    if arg.lower() == 'schnell':
                        self.model = "black-forest-labs/FLUX.1-schnell"
                    elif arg.lower() == 'dev':
                        self.model = "black-forest-labs/FLUX.1-dev"
                    else:
                        self.model = arg
                print(f"🤖 Model: {self.model}")
            else:
                print(f"❓ Unknown command: /{cmd}. Type /help for help.")
            return True

        # Prompt modifications
        if text.startswith('+') and self.current_prompt:
            # Add to prompt
            addition = text[1:].strip()
            new_prompt = f"{self.current_prompt}, {addition}"
            self.generate(new_prompt)
        elif text.startswith('-') and self.current_prompt:
            # Remove from prompt
            removal = text[1:].strip().lower()
            words = self.current_prompt.split(', ')
            words = [w for w in words if removal not in w.lower()]
            new_prompt = ', '.join(words)
            if new_prompt:
                self.generate(new_prompt)
            else:
                print("❌ Can't remove everything. Use ! for a new prompt.")
        elif text.startswith('!'):
            # New prompt entirely
            new_prompt = text[1:].strip()
            if new_prompt:
                self.generate(new_prompt)
            else:
                print("❌ Please provide a prompt after !")
        else:
            # Fresh prompt
            self.generate(text)

        return True


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN environment variable required")
        sys.exit(1)

    # Use local output directory if not in Docker
    default_output = "/app/output/current.png" if os.path.exists("/app") else "output/current.png"
    output = os.environ.get("OUTPUT_PATH", default_output)

    # Ensure output directory exists
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(token, output)

    print("\n🎨 Interactive Image Generator")
    print("Type a prompt to generate, or /help for commands\n")

    try:
        while True:
            try:
                user_input = input("🖼️  > ")
                if not generator.process_input(user_input):
                    break
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
