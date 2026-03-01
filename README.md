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
```bash
bash script/download_ckpts.sh
```


## ⚡ Inference

### Gradio Interactive Demo

Launch the interactive audio editor with a web-based UI:

```bash
./script/launch_gradio_editor.sh
```

#### Demo Usage

<video width="100%" controls>
  <source src="media/smartdj_editor_gradio_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Todo
- [x] Release inference code for SmartDJ-Editor (diffusion editor)
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