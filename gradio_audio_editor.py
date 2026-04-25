"""
Interactive Audio Editor with Gradio UI
This script provides a web-based interface for audio editing using the SmartDJ editor model.
"""

import os
import yaml
import torch
import librosa
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import soundfile as sf

from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler

from editor_model.inference import rescale_noise_cfg
from editor_model.udit import UDiT
from vae_modules.autoencoder_wrapper import Autoencoder


# Global variables to store models
MODELS = None


def create_audio_visualization(audio_path, sr=24000):
    """
        Create spectrogram visualization for an audio file.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate
        
    Returns:
        PIL Image object containing the visualization
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=False)
    
    # If stereo, take first channel for visualization
    if y.ndim > 1:
        y_mono = y[0]
    else:
        y_mono = y
    

    # Create figure with single subplot for spectrogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title('Spectrogram')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def create_audio_visualization_from_array(audio_array, sr=24000):
    """
        Create spectrogram visualization from audio array.
    
    Args:
        audio_array: Audio numpy array (channels, samples) or (samples,)
        sr: Sample rate
        
    Returns:
        PIL Image object containing the visualization
    """
    # If stereo, take first channel for visualization
    if audio_array.ndim > 1:
        y_mono = audio_array[:, 0]
    else:
        y_mono = audio_array
    
    # Create figure with single subplot for spectrogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title('Spectrogram')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


@torch.no_grad()
def load_editor_models(
    diffusion_ckpt,
    autoencoder_ckpt,
    autoencoder_config_path,
    diffusion_config_path,
    token_len=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Load all required models for audio editing."""
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
    ae = Autoencoder(
        ckpt_path=autoencoder_ckpt, 
        model_type="stable_vae", 
        quantization_first=True, 
        config_file=autoencoder_config_path
    ).to(device).eval()

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
def perform_audio_editing(
    audio_file,
    instruction,
    ddim_steps=50,
    eta=0.0,
    guidance_scale=0.0,
    guidance_rescale=0.75,
    progress=gr.Progress()
):
    """
    Perform audio editing based on text instruction.
    
    Args:
        audio_file: Input audio file path
        instruction: Text instruction for editing
        ddim_steps: Number of denoising steps
        eta: DDIM eta parameter
        guidance_scale: Classifier-free guidance scale
        guidance_rescale: Guidance rescale factor
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (output_audio_path, visualization_image, status_message)
    """
    global MODELS
    
    if MODELS is None:
        return None, None, "❌ Models not loaded. Please wait for initialization."
    
    if audio_file is None:
        return None, None, "❌ Please upload an audio file first."
    
    if not instruction or instruction.strip() == "":
        return None, None, "❌ Please provide an editing instruction."
    
    try:
        progress(0, desc="Loading audio...")
        
        device = MODELS["device"]
        tokenizer = MODELS["tokenizer"]
        text_encoder = MODELS["text_encoder"]
        unet = MODELS["unet"]
        autoencoder = MODELS["autoencoder"]
        scheduler = MODELS["scheduler"]
        token_len = MODELS["token_len"]
        sample_rate = 24000

        # Load the audio
        audio_clip, sr = librosa.load(audio_file, sr=sample_rate, mono=False)
        if np.abs(audio_clip).max() > 1:
            audio_clip /= np.abs(audio_clip).max()

        if sr != sample_rate:
            audio_clip = librosa.resample(audio_clip, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
        
        # Make audio binaural
        if audio_clip.ndim == 1:
            audio_clip = np.stack([audio_clip, audio_clip], axis=0)

        audio_clip = torch.tensor(audio_clip, dtype=torch.float32).to(device)
        
        progress(0.1, desc="Encoding audio to latent space...")
        
        # Convert audio signal to latents
        audio_clip = audio_clip.unsqueeze(1).to(device)
        mixture = autoencoder(audio=audio_clip)
        mixture = torch.concat([mixture[0], mixture[1]], dim=0).unsqueeze(0)

        progress(0.2, desc="Processing text instruction...")
        
        # Conditional embedding
        text_inputs = tokenizer(
            [instruction], 
            return_tensors="pt", 
            max_length=token_len, 
            padding="max_length", 
            truncation=True
        )
        
        text_ids = text_inputs.input_ids.to(device)
        text_mask = text_inputs.attention_mask.to(device).bool()
        embedding = text_encoder(input_ids=text_ids, attention_mask=text_mask).last_hidden_state

        # Unconditional embedding
        if guidance_scale > 0:
            uncon_text_inputs = tokenizer(
                ["".ljust(token_len)], 
                return_tensors="pt", 
                max_length=token_len, 
                padding="max_length", 
                truncation=True
            )
            uncon_ids = uncon_text_inputs.input_ids.to(device)
            uncon_mask = uncon_text_inputs.attention_mask.to(device).bool()
            uncond_embedding = text_encoder(input_ids=uncon_ids, attention_mask=uncon_mask).last_hidden_state

        progress(0.25, desc="Starting diffusion process...")
        
        scheduler.set_timesteps(ddim_steps)
        pred = torch.randn_like(mixture, device=device)

        # Denoising loop
        for idx, t in enumerate(scheduler.timesteps):
            progress(0.25 + 0.65 * (idx / len(scheduler.timesteps)), desc=f"Denoising step {idx+1}/{ddim_steps}...")
            
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

        progress(0.9, desc="Decoding audio...")
        
        # Decode the audios
        f_len = pred.shape[1]
        pred = torch.cat([pred[:, :f_len//2], pred[:, f_len//2:]], dim=0)
        audio = autoencoder(embedding=pred)
        audio = torch.cat([audio[:1], audio[1:]], dim=1)

        # Convert to numpy array
        audio_np = audio.squeeze(0).cpu().T.numpy()
        
        progress(0.95, desc="Creating visualization...")
        
        # Create visualization from audio array
        viz_image = create_audio_visualization_from_array(audio_np, sample_rate)
        
        progress(1.0, desc="Complete!")
        
        status_msg = f"✅ Audio editing completed successfully!\n📝 Instruction: {instruction}"
        
        # Return audio as tuple (sample_rate, audio_array) for Gradio
        return (sample_rate, audio_np), viz_image, status_msg
        
    except Exception as e:
        return None, None, f"❌ Error during editing: {str(e)}"


def process_uploaded_audio(audio_file):
    """Process uploaded audio and create visualization."""
    if audio_file is None:
        return None, "Please upload an audio file."
    
    try:
        viz_image = create_audio_visualization(audio_file, sr=24000)
        duration = librosa.get_duration(path=audio_file)
        info = f"✅ Audio loaded successfully!\n⏱️ Duration: {duration:.2f}s"
        return viz_image, info
    except Exception as e:
        return None, f"❌ Error loading audio: {str(e)}"


def get_template_instruction(template_type):
    """Get template instruction based on selected type."""
    templates = {
        "Add Sound": "add the sound of [sound event] at the [left/left front/front/right front/right] with [xx] dB",
        "Remove Sound": "remove the sound of [sound event] at the [left/left front/front/right front/right]",
        "Extract Sound": "extract the sound of [sound event] at the [left/left front/front/right front/right]",
        "Change Volume": "turn [up/down] the volume of [sound event] at [left/left front/front/right front/right] by [xx] dB",
        "Change Direction": "change the sound of [sound event] at the [] to [left/left front/front/right front/right]",
        "Shift Sound Timing": "Shift the sound of [sound event] at the [left/left front/front/right front/right] by [xx] seconds",
        "Add Reveberation": "reverb the sound of [sound event] at the [left/left front/front/right front/right] with reverb level [xx]",
        "Change Timbre": "change the timbre of the sound of [sound event] at the [left/left front/front/right front/right] to be more [bright/dark/warm/cold/muffled]",
        "Custom": ""
    }
    return templates.get(template_type, "")


def create_gradio_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .container {max-width: 1400px; margin: auto;}
    .header {text-align: center; margin-bottom: 2rem;}
    .audio-block {border: 2px dashed #ccc; border-radius: 10px; padding: 20px; margin: 10px 0;}
    .info-box {background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .template-box {background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;}
    """
    
    with gr.Blocks(css=css, title="SmartDJ Audio Editor") as demo:
        
        gr.Markdown("""
        # 🎵 SmartDJ Interactive Audio Editor
        
        Upload your audio, describe the editing you want, and let AI transform your sound!
        """, elem_classes="header")
        
        # Top Row: Original (Left) and Edited (Right) side by side
        with gr.Row():
            # Left Column: Original Audio
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Original Audio")
                with gr.Group(elem_classes="audio-block"):
                    input_audio = gr.Audio(
                        label="Upload or Record Audio",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    input_viz = gr.Image(label="Spectrogram", height=300)
                    input_info = gr.Textbox(label="Audio Info", lines=2, interactive=False)
            
            # Right Column: Edited Audio Output
            with gr.Column(scale=1):
                gr.Markdown("### 🎧 Edited Audio Output")
                with gr.Group(elem_classes="audio-block"):
                    output_audio = gr.Audio(label="Edited Result")
                    output_viz = gr.Image(label="Spectrogram", height=350)
                    output_info = gr.Textbox(label="Status", lines=2, interactive=False)
        
        # Auto-process when audio is uploaded
        input_audio.change(
            fn=process_uploaded_audio,
            inputs=[input_audio],
            outputs=[input_viz, input_info]
        )
        
        # Bottom Section: Editing Controls
        gr.Markdown("### ✏️ Editing Operations")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**Instruction Templates** (Select to auto-fill)")
                template_dropdown = gr.Dropdown(
                    choices=[
                        "Add Sound",
                        "Remove Sound", 
                        "Extract Sound",
                        "Change Volume",
                        "Change Direction",
                        "Shift Sound Timing",
                        "Add Reveberation",
                        "Change Timbre",
                        "Custom"
                    ],
                    value="Custom",
                    label="Template Type",
                    elem_classes="template-box"
                )

                
            with gr.Column(scale=3):
                instruction_input = gr.Textbox(
                    label="Editing Instruction",
                    placeholder="Example: extract the sound of woman screaming and yelling at the right",
                    lines=3,
                    info="Describe what you want to do with the audio"
                )
        
        # Update instruction when template is selected
        template_dropdown.change(
            fn=get_template_instruction,
            inputs=[template_dropdown],
            outputs=[instruction_input]
        )
        
        
        # Advanced Parameters
        with gr.Accordion("⚙️ Advanced Parameters", open=False):
            with gr.Row():
                ddim_steps = gr.Slider(
                    minimum=10, maximum=100, value=50, step=1,
                    label="DDIM Steps",
                    info="More steps = better quality but slower"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0, maximum=10.0, value=0.0, step=0.1,
                    label="Guidance Scale",
                    info="Higher values follow instruction more closely"
                )
            with gr.Row():
                guidance_rescale = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.75, step=0.05,
                    label="Guidance Rescale"
                )
                eta = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                    label="DDIM Eta"
                )
        
        # Process Button
        process_btn = gr.Button("🚀 Process Audio", variant="primary", size="lg")
        
        # Connect the process button
        process_btn.click(
            fn=perform_audio_editing,
            inputs=[
                input_audio,
                instruction_input,
                ddim_steps,
                eta,
                guidance_scale,
                guidance_rescale
            ],
            outputs=[output_audio, output_viz, output_info]
        )
        
        # Examples at the bottom
        gr.Markdown("### 📚 Quick Examples")
        gr.Examples(
            examples=[
                ["./demo_audios/example_001.wav", "extract the sound of woman screaming and laughing at the right"],
                ["./demo_audios/example_001.wav", "add the sound of dog barking at the right with 0 dB"],
                ["./demo_audios/example_001.wav", "remove the sound of woman screaming and laughing at the right"],
                ["./demo_audios/example_002.wav", "change the sound of baby and woman and man speaking at the right to the left"],
                ["./demo_audios/example_002.wav", "remove the sound of sewing machine at the front"],
                ["./demo_audios/example_003.wav", "turn up the sound of a main is speaking at the front by 4 db"],
                ["./demo_audios/example_003.wav", "change the timbre of the sound of a man speaking at the left to be more muffled"],
                ["./demo_audios/example_003.wav", "reverb the sound of man speech at the left with reverb level 5"]
            ],
            inputs=[input_audio, instruction_input, ddim_steps],
            label="Try these examples"
        )
    
    return demo


def main():
    """Main function to initialize and launch the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartDJ Interactive Audio Editor")
    parser.add_argument("--diffusion-ckpt", type=str, 
                       default="./pretrained_models/smartdj_editor.pt",
                       help="Path to diffusion model checkpoint")
    parser.add_argument("--diffusion-config", type=str, 
                       default="./config/diffusion/AudioEdit.yaml",
                       help="Path to diffusion config")
    parser.add_argument("--autoencoder-ckpt", type=str, 
                       default="./pretrained_models/audio_vae.pt",
                       help="Path to autoencoder checkpoint")
    parser.add_argument("--autoencoder-config", type=str, 
                       default="./config/vae/24k_mono_latent64.json",
                       help="Path to autoencoder config")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the server on")
    parser.add_argument("--share", action="store_true",
                       help="Create a public link")
    
    args = parser.parse_args()
    
    # Load models
    global MODELS
    print("🔄 Loading models... This may take a few minutes.")
    
    try:
        MODELS = load_editor_models(
            diffusion_ckpt=args.diffusion_ckpt,
            autoencoder_ckpt=args.autoencoder_ckpt,
            diffusion_config_path=args.diffusion_config,
            autoencoder_config_path=args.autoencoder_config
        )
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please check your checkpoint paths and try again.")
        return
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    print(f"\n🚀 Launching SmartDJ Audio Editor...")
    print(f"📍 Server will run on http://{args.host}:{args.port}")
    if args.share:
        print("🌐 Public link will be generated...")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()