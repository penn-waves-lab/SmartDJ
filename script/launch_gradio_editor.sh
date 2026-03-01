#!/bin/bash

# Launch script for SmartDJ Gradio Audio Editor
# This script helps you quickly launch the interactive audio editor

echo "🎵 SmartDJ Interactive Audio Editor"
echo "===================================="
echo ""


# Launch the editor
echo ""
echo "🚀 Launching Gradio Audio Editor..."
echo ""
echo ""

# Parse arguments
EXTRA_ARGS=""
if [ "$1" == "--share" ] || [ "$2" == "--share" ] || [ "$3" == "--share" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --share"
fi

# Run the editor
python3 gradio_audio_editor.py \
    --diffusion-ckpt ./pretrained_models/smartdj_editor.pt  \
    --diffusion-config ./config/diffusion/AudioEdit.yaml \
    --autoencoder-ckpt ./pretrained_models/24k_mono_latent64.ckpt \
    --autoencoder-config ./config/vae/24k_mono_latent64.json \
    --share