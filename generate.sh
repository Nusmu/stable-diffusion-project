#!/bin/bash
# Simple wrapper to generate images with Stable Diffusion
# Usage: ./generate.sh "your prompt here" [options]
#        ./generate.sh --xl "your prompt here" [options]  # For SDXL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory if it doesn't exist
mkdir -p output

# Check for mode flags
USE_SDXL=false
USE_API=false
if [[ "$1" == "--xl" ]]; then
    USE_SDXL=true
    shift
elif [[ "$1" == "--api" ]]; then
    USE_API=true
    shift
fi

# Show help if no arguments
if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Stable Diffusion Image Generator"
    echo ""
    echo "Usage:"
    echo "  ./generate.sh \"your prompt\" [options]        # SD 1.5 (512x512, local GPU)"
    echo "  ./generate.sh --xl \"your prompt\" [options]   # SDXL (1024x1024, local GPU)"
    echo "  ./generate.sh --api \"your prompt\" [options]  # SDXL via HuggingFace API (no GPU needed)"
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
if [[ "$USE_API" == true ]]; then
    echo "Using HuggingFace Inference API (no local GPU)..."
    docker compose run --rm api "$@" -o "/app/output/output_api.png"
elif [[ "$USE_SDXL" == true ]]; then
    echo "Using SDXL (local GPU)..."
    docker compose run --rm sdxl "$@"
else
    echo "Using Stable Diffusion 1.5..."
    docker compose run --rm sd "$@"
fi

echo ""
echo "Done! Check the ./output directory for your image."
