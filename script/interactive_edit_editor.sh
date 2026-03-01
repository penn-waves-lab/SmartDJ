# ---------- Interactive editing demos --------
python3 editor_demo.py \
    --diffusion-ckpt './pretrained_models/smartdj_editor.pt' \
    --diffusion-config './config/diffusion/AudioEdit.yaml' \
    --autoencoder-ckpt './pretrained_models/24k_mono_latent64.ckpt' \
    --autoencoder-config './config/vae/24k_mono_latent64.json' \
    --guidance-scale 4 \
    --eta 0.8 \
    --guidance-rescale 0.8 \
    --ddim-steps 50
