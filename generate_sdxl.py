#!/usr/bin/env python3
"""
SDXL-Turbo Image Generator

Fast, high-quality images with SDXL-Turbo (1-4 steps).
Requires ~8GB VRAM.
"""

import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_id: str, device: str):
    """Load the SDXL-Turbo pipeline."""
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        variant="fp16" if device != "cpu" else None,
    )

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int = None,
):
    """Generate an image from a text prompt."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if guidance_scale > 0 else None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="Generate images with SDXL-Turbo")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("-n", "--negative", type=str,
                        default="",
                        help="Negative prompt (only used with guidance > 0)")
    parser.add_argument("-o", "--output", type=str, default="output_xl.png",
                        help="Output filename")
    parser.add_argument("-m", "--model", type=str,
                        default="stabilityai/sdxl-turbo",
                        help="SDXL model ID")
    parser.add_argument("-W", "--width", type=int, default=512, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=512, help="Image height")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Inference steps (1-4 for turbo)")
    parser.add_argument("-g", "--guidance", type=float, default=0.0, help="Guidance scale (0.0 for turbo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    pipe = load_pipeline(args.model, device)

    print(f"Generating image for prompt: '{args.prompt}'")
    image = generate_image(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    output_path = Path(args.output)
    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
