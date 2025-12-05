#!/usr/bin/env python3
"""
Stable Diffusion via Hugging Face Inference API

No local GPU required - runs on HuggingFace servers.
"""

import argparse
import os
import requests
from pathlib import Path


def generate_image(
    prompt: str,
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    negative_prompt: str = "",
    token: str = None,
):
    """Generate an image using HuggingFace Inference API."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable or --token required")

    API_URL = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    payload = {"inputs": prompt}
    if negative_prompt:
        payload["parameters"] = {"negative_prompt": negative_prompt}

    print(f"Sending request to {model}...")
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        error = response.json() if response.headers.get('content-type') == 'application/json' else response.text
        raise Exception(f"API error {response.status_code}: {error}")

    return response.content


def main():
    parser = argparse.ArgumentParser(description="Generate images via HuggingFace API")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("-n", "--negative", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("-o", "--output", type=str, default="output_api.png",
                        help="Output filename")
    parser.add_argument("-m", "--model", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Model ID")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token (or set HF_TOKEN env var)")

    args = parser.parse_args()

    image_bytes = generate_image(
        prompt=args.prompt,
        model=args.model,
        negative_prompt=args.negative,
        token=args.token,
    )

    output_path = Path(args.output)
    output_path.write_bytes(image_bytes)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
