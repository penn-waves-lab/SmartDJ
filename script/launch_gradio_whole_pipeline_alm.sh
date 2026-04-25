#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export GEMINI_API_KEY=""  # Replace with your own API key
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
export GEMINI_BASE_URL="${GEMINI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "❌ GEMINI_API_KEY is not set."
    echo "Set it first, for example: export GEMINI_API_KEY='your_api_key'"
    exit 1
fi


python3 gradio_whole_pipeline_alm.py \
        --gemini-model "${GEMINI_MODEL}" \
        --gemini-base-url "${GEMINI_BASE_URL}" \
        --gemini-api-key "${GEMINI_API_KEY}" \
    --diffusion-ckpt ./pretrained_models/smartdj_editor.pt \
    --diffusion-config ./config/diffusion/AudioEdit.yaml \
    --autoencoder-ckpt ./pretrained_models/24k_mono_latent64.ckpt \
    --autoencoder-config ./config/vae/24k_mono_latent64.json \
    --add-model-ckpt ./pretrained_models/add_model.pt \
    --add-model-vae-ckpt ./pretrained_models/audio-vae.ckpt \
    --share
