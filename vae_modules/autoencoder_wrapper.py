import torch
import torch.nn as nn
from .dac import DAC
from scipy.io import wavfile
import librosa
import numpy as np
from stable_vae import load_vae

class Autoencoder(nn.Module):
    def __init__(self, ckpt_path, model_type='stable_vae', quantization_first=True, config_file=None):
        super(Autoencoder, self).__init__()
        self.model_type = model_type
        if self.model_type == 'dac':
            model = DAC.load(ckpt_path)
        elif self.model_type == 'stable_vae':
            model = load_vae(ckpt_path, config_file=config_file)
        elif self.model_type == 'stable_vae_stereo':
            model = load_vae(ckpt_path, config_file=config_file)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")
        self.ae = model.eval()
        self.quantization_first = quantization_first
        print(f'Autoencoder quantization first mode: {quantization_first}')

    @torch.no_grad()
    def forward(self, audio=None, embedding=None):
        if self.model_type == 'dac':
            return self.process_dac(audio, embedding)
        elif self.model_type == 'encodec':
            return self.process_encodec(audio, embedding)
        elif self.model_type == 'stable_vae' or self.model_type == 'stable_vae_stereo':
            return self.process_stable_vae(audio, embedding)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")

    def process_dac(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z, *_ = self.ae.quantizer(z, None)
            return z
        elif embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z, *_ = self.ae.quantizer(z, None)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_encodec(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                code = self.ae.quantizer.encode(z)
                z = self.ae.quantizer.decode(code)
            return z
        elif embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                code = self.ae.quantizer.encode(z)
                z = self.ae.quantizer.decode(code)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_stable_vae(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z = self.ae.bottleneck.encode(z)
            return z
        if embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z = self.ae.bottleneck.encode(z)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")
        
@torch.no_grad()
def test_autoencoder(autoencoder, input_path, save_path, target_sr=24000, model_stereo=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    waveform, ori_fs = librosa.load(input_path, sr=None, mono=False)
    waveform = torch.tensor(waveform).to(device=device)

    data_stereo = True if len(waveform.shape) == 2 else False

    if ori_fs != target_sr:
        waveform = librosa.resample(waveform, orig_sr=ori_fs, target_sr=target_sr)
        waveform = torch.tensor(waveform).to(device=device)

    if not model_stereo and data_stereo:
        left = waveform[[0],:]
        right = waveform[[1],:]

        left_z = autoencoder.process_stable_vae(audio=left)
        recon_left = autoencoder.process_stable_vae(embedding=left_z)

        right_z = autoencoder.process_stable_vae(audio=right)
        recon_right = autoencoder.process_stable_vae(embedding=right_z)

        out_audio = torch.concat([recon_left.squeeze(0), recon_right.squeeze(0)], dim=0).T.cpu().numpy()
        wavfile.write('recond_music.wav', target_sr, (out_audio * 32767).astype("int16"))


    else:
        latent = autoencoder.process_stable_vae(audio=waveform)
        recon = autoencoder.process_stable_vae(embedding=latent)
        
        recon = np.clip(recon.cpu().squeeze().numpy().T, -0.999, 0.999)
        wavfile.write('recond_music.wav', target_sr, (recon * 32767).astype("int16"))

