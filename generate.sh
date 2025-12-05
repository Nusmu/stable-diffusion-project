#!/bin/bash
# Simple wrapper to generate images with Stable Diffusion
# Usage: ./generate.sh "your prompt here" [options]
#        ./generate.sh --xl "your prompt here" [options]  # For SDXL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory if it doesn't exist
mkdir -p output

# Check for --xl flag
USE_SDXL=false
if [[ "$1" == "--xl" ]]; then
    USE_SDXL=true
    shift
fi

# Show help if no arguments
if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Stable Diffusion Image Generator"
    echo ""
    echo "Usage:"
    echo "  ./generate.sh \"your prompt\" [options]        # SD 1.5 (512x512)"
    echo "  ./generate.sh --xl \"your prompt\" [options]   # SDXL + Refiner (1024x1024, best quality)"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE    Output filename (default: output.png)"
    echo "  -n, --negative TEXT  Negative prompt"
    echo "  -s, --steps NUM      Inference steps (more = better quality, slower)"
    echo "  -g, --guidance NUM   Guidance scale (how closely to follow prompt)"
    echo "  --seed NUM           Random seed for reproducibility"
    echo "  -W, --width NUM      Image width"
    echo "  -H, --height NUM     Image height"
    echo ""
    echo "Examples:"
    echo "  ./generate.sh \"a cat astronaut on mars\""
    echo "  ./generate.sh \"sunset over mountains\" -o sunset.png"
    echo "  ./generate.sh --xl \"photorealistic portrait\"  # best quality"
    exit 0
fi

# Build image if needed
if ! docker images | grep -q "stable-diffusion.*latest"; then
    echo "Building Docker image (first run)..."
    docker compose build
fi

# Run the appropriate service
if [[ "$USE_SDXL" == true ]]; then
    echo "Using SDXL + Refiner (best quality)..."
    docker compose run --rm sdxl "$@" -o "/app/output/$(basename "${2:-output_xl.png}")"
else
    echo "Using Stable Diffusion 1.5..."
    docker compose run --rm sd "$@" -o "/app/output/$(basename "${2:-output.png}")"
fi

echo ""
echo "Done! Check the ./output directory for your image."
