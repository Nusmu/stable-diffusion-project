#!/usr/bin/env python3
"""
SDXL Image Generator with Refiner

Highest quality images using SDXL base + refiner pipeline.
Requires ~12-14GB VRAM (16GB recommended).
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipelines(base_model: str, refiner_model: str, device: str, use_refiner: bool = True):
    """Load the SDXL base and optionally refiner pipelines."""
    dtype = torch.float16 if device != "cpu" else torch.float32

    # Load base model
    base = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device != "cpu" else None,
    )
    base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)

    if device == "cuda":
        # Use CPU offload to fit both models in VRAM
        base.enable_model_cpu_offload()
    else:
        base = base.to(device)

    refiner = None
    if use_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_model,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if device != "cpu" else None,
        )

        if device == "cuda":
            refiner.enable_model_cpu_offload()
        else:
            refiner = refiner.to(device)

    return base, refiner


def generate_image(
    base,
    refiner,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 40,
    refiner_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    """Generate an image using base + refiner pipeline."""
    generator = None
    if seed is not None:
        # Use CPU generator for compatibility with CPU offload
        generator = torch.Generator("cpu").manual_seed(seed)

    # Calculate denoising split for base/refiner
    high_noise_frac = 0.8  # Base handles 80% of denoising

    # Generate with base model
    if refiner is not None:
        # Two-stage: base outputs latents, refiner finishes
        image = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        # Refine the image
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=refiner_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            denoising_start=high_noise_frac,
        ).images[0]
    else:
        # Single-stage: base only
        image = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate images with SDXL + Refiner")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("-n", "--negative", type=str,
                        default="blurry, bad quality, distorted, ugly, deformed, low resolution, pixelated",
                        help="Negative prompt")
    parser.add_argument("-o", "--output", type=str, default="/app/output/output_xl.png",
                        help="Output filename")
    parser.add_argument("-m", "--model", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="SDXL base model ID")
    parser.add_argument("--refiner", type=str,
                        default="stabilityai/stable-diffusion-xl-refiner-1.0",
                        help="SDXL refiner model ID")
    parser.add_argument("--no-refiner", action="store_true",
                        help="Disable refiner (faster, less quality)")
    parser.add_argument("-W", "--width", type=int, default=1024, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=1024, help="Image height")
    parser.add_argument("-s", "--steps", type=int, default=40, help="Base inference steps")
    parser.add_argument("--refiner-steps", type=int, default=20, help="Refiner inference steps")
    parser.add_argument("-g", "--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    use_refiner = not args.no_refiner
    print(f"Loading SDXL base: {args.model}")
    if use_refiner:
        print(f"Loading SDXL refiner: {args.refiner}")

    base, refiner = load_pipelines(args.model, args.refiner, device, use_refiner)

    print(f"Generating image for prompt: '{args.prompt}'")
    image = generate_image(
        base,
        refiner,
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        steps=args.steps,
        refiner_steps=args.refiner_steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    output_path = Path(args.output)
    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
