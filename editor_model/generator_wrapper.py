import os
import torch
import torch.nn as nn
import random

import numpy as np
from typing import Optional, Tuple

from accelerate import Accelerator
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "inference"))


from vae_modules.autoencoder_wrapper import Autoencoder
from editor_model.udit import UDiT
from editor_model.inference import rescale_noise_cfg
import yaml

MAX_SEED = np.iinfo(np.int32).max


def load_yaml_with_includes(yaml_file):
    def loader_with_include(loader, node):
        # Load the included file
        include_path = os.path.join(os.path.dirname(yaml_file), loader.construct_scalar(node))
        with open(include_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    yaml.add_constructor('!include', loader_with_include, Loader=yaml.FullLoader)

    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    

@torch.no_grad()
def inference(autoencoder, unet, gt, gt_mask,
              tokenizer, text_encoder,
              params, noise_scheduler,
              text_raw, neg_text=None,
              audio_frames=500,
              guidance_scale=3, guidance_rescale=0.0,
              ddim_steps=50, eta=1, random_seed=2024,
              device='cuda',
              ):
    if neg_text is None:
        neg_text = [""]
    if tokenizer is not None:
        text_batch = tokenizer(text_raw,
                               max_length=params['text_encoder']['max_length'],
                               padding="max_length", truncation=True, return_tensors="pt")
        text, text_mask = text_batch.input_ids.to(device), text_batch.attention_mask.to(device).bool()
        text = text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state

        uncond_text_batch = tokenizer(neg_text,
                                      max_length=params['text_encoder']['max_length'],
                                      padding="max_length", truncation=True, return_tensors="pt")
        uncond_text, uncond_text_mask = uncond_text_batch.input_ids.to(device), uncond_text_batch.attention_mask.to(device).bool()
        uncond_text = text_encoder(input_ids=uncond_text,
                                   attention_mask=uncond_text_mask).last_hidden_state
    else:
        text, text_mask = None, None
        guidance_scale = None

    codec_dim = params['model']['out_chans']
    unet.eval()

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()

    noise_scheduler.set_timesteps(ddim_steps)

    # init noise
    noise = torch.randn((1, codec_dim, audio_frames), generator=generator, device=device)
    latents = noise

    for t in noise_scheduler.timesteps:
        latents = noise_scheduler.scale_model_input(latents, t)

        if guidance_scale:

            latents_combined = torch.cat([latents, latents], dim=0)
            text_combined = torch.cat([text, uncond_text], dim=0)
            text_mask_combined = torch.cat([text_mask, uncond_text_mask], dim=0)
            
            if gt is not None:
                gt_combined = torch.cat([gt, gt], dim=0)
                gt_mask_combined = torch.cat([gt_mask, gt_mask], dim=0)
            else:
                gt_combined = None
                gt_mask_combined = None
            
            output_combined, _ = unet(latents_combined, t, text_combined, context_mask=text_mask_combined, 
                                      cls_token=None, gt=gt_combined, mae_mask_infer=gt_mask_combined)
            output_text, output_uncond = torch.chunk(output_combined, 2, dim=0)

            output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
            if guidance_rescale > 0.0:
                output_pred = rescale_noise_cfg(output_pred, output_text,
                                                guidance_rescale=guidance_rescale)
        else:
            output_pred, mae_mask = unet(latents, t, text)

        latents = noise_scheduler.step(model_output=output_pred, timestep=t, 
                                       sample=latents,
                                       eta=eta, generator=generator).prev_sample

    pred = scale_shift_re(latents, params['autoencoder']['scale'],
                          params['autoencoder']['shift'])
    if gt is not None:
        pred[~gt_mask] = gt[~gt_mask]
    pred_wav = autoencoder(embedding=pred)
    return pred_wav


def scale_shift_re(x, scale, shift):
    return (x/scale) - shift

class AddModelWrapper:
    def __init__(self, ckpt_path=None, vae_path=None, device='cuda'):
        self.device = device
        config_name = "config/diffusion/AddModel.yaml"


        (self.autoencoder, self.unet, self.tokenizer,
         self.text_encoder, self.noise_scheduler, self.params) = self.load_models(config_name, ckpt_path, vae_path, device)


    # Load model and configs
    def load_models(self, config_name, ckpt_path, vae_path, device):
        params = load_yaml_with_includes(config_name)
        
        # Load codec model
        autoencoder = Autoencoder(ckpt_path=vae_path,model_type=params['autoencoder']['name'], quantization_first=params['autoencoder']['q_first'], config_file="./config/vae/add_vae_config.json").to(device)
        
        autoencoder.eval()

        # Load text encoder
        tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
        text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model']).to(device)
        text_encoder.eval()

        # Load main U-Net model
        unet = MaskDiT(**params['model']).to(device)
        
        unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
        unet.eval()

        if device == 'cuda':
            accelerator = Accelerator(mixed_precision="fp16")
            unet = accelerator.prepare(unet)

        # Load noise scheduler
        noise_scheduler = DDIMScheduler(**params['diff'])

        latents = torch.randn((1, 128, 128), device=device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        _ = noise_scheduler.add_noise(latents, noise, timesteps)

        return autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params

    def generate_adding_audio(self, text, length=10,
                       guidance_scale=5, guidance_rescale=0.75, ddim_steps=100, eta=1,
                       random_seed=None, randomize_seed=False):
        neg_text = None
        length = length * self.params['autoencoder']['latent_sr']

        gt, gt_mask = None, None

        if text == '':
            guidance_scale = None
            print('empyt input')

        if randomize_seed:
            random_seed = random.randint(0, MAX_SEED)

        pred = inference(self.autoencoder, self.unet,
                         gt, gt_mask,
                         self.tokenizer, self.text_encoder,
                         self.params, self.noise_scheduler,
                         text, neg_text,
                         length,
                         guidance_scale, guidance_rescale,
                         ddim_steps, eta, random_seed,
                         self.device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        return self.params['autoencoder']['sr'], pred




class MaskDiT(nn.Module):
    def __init__(self, mae=False, mae_prob=0.5, mask_ratio=[0.25, 1.0], mask_span=10, **kwargs):
        super().__init__()
        self.model = UDiT(**kwargs)
        self.mae = mae
        if self.mae:
            out_channel = kwargs.pop('out_chans', None)
            self.mask_embed = nn.Parameter(torch.zeros((out_channel)))
            self.mae_prob = mae_prob
            self.mask_ratio = mask_ratio
            self.mask_span = mask_span

    def forward(self, x, timesteps, context,
                x_mask=None, context_mask=None, cls_token=None,
                gt=None, mae_mask_infer=None,
                forward_model=True):
        # todo: handle controlnet inside
        mae_mask = torch.ones_like(x)
        
        if self.mae:
            B, D, L = x.shape
            gt = self.mask_embed.view(1, D, 1).expand_as(x)
            x = torch.cat([x, gt, mae_mask[:, 0:1, :]], dim=1)

        if forward_model:
            x = self.model(x=x, timesteps=timesteps, context=context, x_mask=x_mask, context_mask=context_mask, cls_token=cls_token)

        return x, mae_mask
    
    
def create_spatial_audio(audio, dir=None, sr=24000, ear_distance=0.2, sound_speed=340):
    out = np.tile(audio[:, np.newaxis], (1,2))

    if dir is None or dir == 'front':
        return out
    
    elif dir == 'left':
        # delay the right channel
        delay_idx = int(ear_distance/sound_speed*sr)
        out[:,1] = np.pad(out[:,1]*0.6, (delay_idx, 0), mode='constant')[:out.shape[0]]

    elif dir == 'right':
        # delay the left channel
        delay_idx = int(ear_distance/sound_speed*sr)
        out[:,0] = np.pad(out[:,0]*0.6, (delay_idx, 0), mode='constant')[:out.shape[0]]

    elif dir == 'left front':
        delay_idx = int((ear_distance/sound_speed*sr)*0.7)
        out[:,1] = np.pad(out[:,1]*0.8, (delay_idx, 0), mode='constant')[:out.shape[0]]

    elif dir == 'right front':
        delay_idx = int((ear_distance/sound_speed*sr)*0.7)
        out[:,0] = np.pad(out[:,0]*0.8, (delay_idx, 0), mode='constant')[:out.shape[0]]

    return out

        

if __name__ == "__main__":
    import soundfile as sf

    addmodel = AddModelWrapper(ckpt_path='./pretrained_models/add_model.pt', vae_path='./pretrained_models/audio-vae.ckpt')

    add_prompt = "Child is laughing"

    sr, add_audio = addmodel.generate_adding_audio(add_prompt)
    
    # save the generated audio
    output_path = "add_output.wav"
    sf.write(output_path, add_audio, samplerate=sr)