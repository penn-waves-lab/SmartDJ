<h1 align="center">SmartDJ: Declarative Audio Editing with Audio Langugae Model</h1>
<p align="center"><a href="https://arxiv.org/abs/2509.21625"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://zitonglan.github.io/project/smartdj/smartdj'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>



## 📦 Installation
Clone the repository:
```
git clone https://github.com/penn-waves-lab/SmartDJ.git
```
Install the dependencies:
```
cd SmartDJ
pip install -r requirements.txt
```

## 🤖 Pretrained Models
Download the pretrained SmartDJ-Editor here for interactive editing

```bash
bash script/download_ckpts.sh
```


## ⚡ Inference

### Gradio Interactive Demo

Launch the interactive audio editor with a web-based UI:

```bash
bash ./script/launch_gradio_editor.sh
```

#### Demo Usage

https://github.com/user-attachments/assets/c2b364b7-3396-4e95-9807-96ee475fd1be

### Command Line Interactive Demo for Audio Editor
Alternatively, you can also use the command line interactive demo for the SmartDJ-Editor

```bash
bash ./script/interactive_edit_editor.sh
```

<strong>🛠️ Available Commands</strong>

We support the following editing commands.  
**Spatial locations:** {`left` | `left front` | `front` | `right front` | `right`}

| Operation | Command |
|----------|--------|
| ❌ **Remove Sound** | `remove the sound of [sound event] at the {spacial location}` |
| ➕ **Add Sound** | `add the sound of [sound event] at the {spacial location} with [xx] dB` |
| 🎯 **Extract Sound** | `extract the sound of [sound event] at the {spacial location}` |
| 🔊 **Change Volume** | `turn {up \| down} the volume of [sound event] at {spacial location} by [xx] dB` |
| 🧭 **Change Direction** | `change the sound of [sound event] at [original position] to {spacial location}` |
| ⏱️ **Shift Sound Timing** | `shift the sound of [sound event] at the {spacial location} by [xx] seconds` |
| 🌊 **Add Reverberation** | `reverb the sound of [sound event] at the {spacial location} with reverb level [xx]` |
| 🎨 **Change Timbre** | `change the timbre of the sound of [sound event] at the {spacial location} to be more {bright \| dark \| warm \| cold \| muffled}` |


## Todo
- [x] Release inference code and weight for SmartDJ-Editor (diffusion editor)
- [ ] Release inference code for SmartDJ-Planer (ALM planer)
- [ ] Release dataset synthesis pipeline


## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{lan2025guiding,
  title={Guiding audio editing with audio language model},
  author={Lan, Zitong and Hao, Yiduo and Zhao, Mingmin},
  journal={arXiv preprint arXiv:2509.21625},
  year={2025}
}
```

