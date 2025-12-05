#!/usr/bin/env python3
"""
Stable Diffusion Image Generator

Simple script to generate images using Stable Diffusion models.
Models are downloaded automatically on first run.
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def load_pipeline(model_id: str, device: str):
    """Load the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        safety_checker=None,  # Disable for faster inference
    )

    # Use DPM++ scheduler for better quality with fewer steps
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    # Enable memory optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()
        # Uncomment below if you have low VRAM (uses more RAM but less VRAM)
        # pipe.enable_sequential_cpu_offload()

    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    """Generate an image from a text prompt."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("-n", "--negative", type=str, default="blurry, bad quality, distorted",
                        help="Negative prompt (things to avoid)")
    parser.add_argument("-o", "--output", type=str, default="output.png",
                        help="Output filename")
    parser.add_argument("-m", "--model", type=str,
                        default="stabilityai/stable-diffusion-2-1-base",
                        help="Model ID from Hugging Face")
    parser.add_argument("-W", "--width", type=int, default=512, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=512, help="Image height")
    parser.add_argument("-s", "--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("-g", "--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

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
