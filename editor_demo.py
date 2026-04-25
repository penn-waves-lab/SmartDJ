import os
import yaml
import torch
import librosa
import numpy as np
import shutil   
from tqdm import tqdm
import time

from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler

from utils.audio_utils import save_audio
from editor_model.inference import rescale_noise_cfg
from editor_model.udit import UDiT
from vae_modules.autoencoder_wrapper import Autoencoder


@torch.no_grad()
def load_editor_models(
    diffusion_ckpt,
    autoencoder_ckpt,
    autoencoder_config_path,
    diffusion_config_path,
    token_len=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Load config
    with open(diffusion_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer + text encoder
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device).eval()

    # Load diffusion model
    unet = UDiT(**config["model"]).to(device).eval()
    state_dict = torch.load(diffusion_ckpt, map_location="cpu")["model"]
    unet.load_state_dict(state_dict, strict=False)

    # Load autoencoder
    ae = Autoencoder(ckpt_path=autoencoder_ckpt, model_type="stable_vae", quantization_first=True, config_file=autoencoder_config_path).to(device).eval()

    # Load scheduler
    scheduler = DDIMScheduler(**config["ddim"]["diffusers"])

    return {
        "unet": unet,
        "autoencoder": ae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "scheduler": scheduler,
        "device": device,
        "config": config,
        "token_len": token_len,
    }


@torch.no_grad()
def model_inference(
    target_path,
    instruction,
    models,
    output_path="edited_output.wav",
    sample_rate=24000,
    ddim_steps=50,
    eta=0.0,
    guidance_scale=0.0,
    guidance_rescale=0.75,
):
    device = models["device"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    autoencoder = models["autoencoder"]
    scheduler = models["scheduler"]
    token_len = models["token_len"]


    # Load the audio
    audio_clip, sr = librosa.load(target_path, sr=sample_rate, mono=False)
    if np.abs(audio_clip).max() > 1:
        audio_clip /= np.abs(audio_clip).max()

    if sr != sample_rate:
        audio_clip = librosa.resample(audio_clip, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    # Make audio binaural
    if audio_clip.ndim == 1:
        audio_clip = np.stack([audio_clip, audio_clip], axis=0)  # [2, n_len]

    audio_clip = torch.tensor(audio_clip, dtype=torch.float32).to(device)
    
    # Convert audio signal to latents
    audio_clip = audio_clip.unsqueeze(1).to(device) # [2,1,n_len]
    mixture = autoencoder(audio=audio_clip) # [2,1,n_len]
    mixture = torch.concat([mixture[0], mixture[1]], dim=0).unsqueeze(0) # [1, 2*feature_len, tokens]


    # Conditional embedding
    text_inputs = tokenizer([instruction], return_tensors="pt", max_length=token_len, padding="max_length", truncation=True)
    
    text_ids = text_inputs.input_ids.to(device)
    text_mask = text_inputs.attention_mask.to(device).bool()
    embedding = text_encoder(input_ids=text_ids, attention_mask=text_mask).last_hidden_state

    # Unconditional embedding
    if guidance_scale > 0:
        uncon_text_inputs = tokenizer(["".ljust(token_len)], return_tensors="pt", max_length=token_len, padding="max_length", truncation=True)
        uncon_ids = uncon_text_inputs.input_ids.to(device)
        uncon_mask = uncon_text_inputs.attention_mask.to(device).bool()
        uncond_embedding = text_encoder(input_ids=uncon_ids, attention_mask=uncon_mask).last_hidden_state


    scheduler.set_timesteps(ddim_steps)
    pred = torch.randn_like(mixture, device=device)

    for t in tqdm(scheduler.timesteps):
        pred_input = scheduler.scale_model_input(pred, t)

        if guidance_scale > 0:
            pred_concat = torch.cat([pred_input, pred_input], dim=0)
            mix_concat = torch.cat([mixture, mixture], dim=0)
            model_input = torch.cat([pred_concat, mix_concat], dim=1)
            emb_concat = torch.cat([embedding, uncond_embedding], dim=0)

            output = unet(x=model_input, timesteps=t, context=emb_concat)

            pos, neg = torch.chunk(output, 2, dim=0)
            model_output = neg + guidance_scale * (pos - neg)
            
            if guidance_rescale > 0:
                model_output = rescale_noise_cfg(model_output, pos, guidance_rescale=guidance_rescale)
        
        else:
            model_input = torch.cat([pred_input, mixture], dim=1)
            model_output = unet(x=model_input, timesteps=t, context=embedding)

        pred = scheduler.step(model_output=model_output, timestep=t, sample=pred, eta=eta).prev_sample

    # Decode the audios
    f_len = pred.shape[1]
    pred = torch.cat([pred[:, :f_len//2], pred[:, f_len//2:]], dim=0)
    audio = autoencoder(embedding=pred)
    audio = torch.cat([audio[:1], audio[1:]], dim=1)

    base_name = os.path.basename(output_path)
    output_dir = os.path.dirname(output_path)
    
    shutil.copyfile(target_path, os.path.join(output_dir, "ori_" + base_name))

    true_output = os.path.join(output_dir, "edit_" + base_name)
    # Save the original and edited audio
    save_audio(true_output, sample_rate, audio.squeeze(0).cpu().T)
    print(f"Edited audio saved: {true_output}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-ckpt", type=str, default="./pretrained_models/smartdj_editor.pt")
    parser.add_argument("--diffusion-config", type=str, default="./config/diffusion/AudioEdit.yaml")

    parser.add_argument("--autoencoder-ckpt", type=str, default="./pretrained_models/24k_mono_latent64.ckpt")
    parser.add_argument("--autoencoder-config", type=str, default="../config/vae/24k_mono_latent64.json")

    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.75)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--ddim-steps", type=int, default=50)

    parser.add_argument("--output", type=str, default="./examples")

    args = parser.parse_args()

    # Load model
    models = load_editor_models(
        diffusion_ckpt=args.diffusion_ckpt,
        autoencoder_ckpt=args.autoencoder_ckpt,
        diffusion_config_path=args.diffusion_config,
        autoencoder_config_path=args.autoencoder_config
    )


    if os.path.exists(args.output) is False: # create output directory if not exist
        os.makedirs(args.output)

    # Interactive loop
    while True:

        print("\n--- Audio Editing ---")
        
        target_path = input("Enter mixture audio file path, e,g, ./demo_audios/example_001.wav, enter quit to quit the program): ").strip()
        if target_path == "":
            print("Please enter a valid file path.")
            continue

        if target_path.lower() in ["quit", "q"]:
            print("Quitting the program.")
            break

        prompt = input("Enter your editing instruction, e.g. extract the sound of woman screaming and yelling at the right: ").strip()
        if prompt == "":
            print("Please enter a valid prompt.")
            continue

        out_name = input("Output file name (e.g., output.wav is default): ").strip()

        if out_name.lower() == "":
            out_name = "output.wav"

        model_inference(
            target_path=target_path,
            instruction=prompt,
            models=models,
            output_path=os.path.join(args.output, out_name),
            guidance_scale=args.guidance_scale,
            guidance_rescale=args.guidance_rescale,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
        )

    print("Exiting interactive mode.")