
<p align="center">
<!--   <h1 align="center"><img height="100" src="https://github.com/imlixinyang/director3d-page/raw/master/assets/icon.ico"></h1> -->
  <h1 align="center">🎥 <b>Director3D</b>: Real-world Camera Trajectory and 3D Scene Generation from Text</h1>
  <p align="center">
        <a href="https://arxiv.org/pdf/2406.17601"><img src='https://img.shields.io/badge/arXiv-Director3D-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://imlixinyang.github.io/director3d-page'><img src='https://img.shields.io/badge/Project_Page-Director3D-green' alt='Project Page'></a>
        <a href='https://colab.research.google.com/drive/1LtnxgBU7k4gyymOWuonpOxjatdJ7AI8z?usp=sharing'><img src='https://img.shields.io/badge/Colab_Demo-Director3D-yellow?logo=googlecolab' alt='Project Page'></a>
  </p>

<img src='assets/pipeline.gif'>

**⭐ Key components of Director3D**:

- A trajectory diffusion model for generating suquential camera intrinsics & extrinsics given texts.
- A 3DGS-driven multi-view latent diffusion model for generating coarse 3DGS given cameras and texts in 20 seconds.
- A more advanced SDS loss, named SDS++, for refining coarse 3DGS to real-world visual quality in 5 minutes.

**🔥 News**:

- 🥰 Check out our new gradio demo by simply running ```python app.py```.

- 🆓 Try out Director3D for free with our [**Google Colab Demo**](https://colab.research.google.com/drive/1LtnxgBU7k4gyymOWuonpOxjatdJ7AI8z?usp=sharing).

- 😊 Our paper is accepted by NeurIPS 2024.

## 📖 Generation Results

❗ All videos are rendered with generated camera trajectories and 3D Gaussians, the only inputs are text prompts!

https://github.com/imlixinyang/Director3D/assets/26456614/b4e7d910-e3fd-4d32-895b-e35b837bc9a1


👀 See more than 200 examples in our [**Gallery**](https://imlixinyang.github.io/director3d-page/gallery_0.html).


## 🔧 Installation
- create a new conda enviroment

```
conda create -n director3d python=3.9
conda activate director3d
```

- install pytorch (or use your own if it is compatible with ```xformers```)
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
- install ```xformers``` for momory-efficient attention
```
conda install xformers -c xformers
```
- install ```pip``` packages
```
pip install kiui scipy opencv-python-headless kornia omegaconf imageio imageio-ffmpeg  seaborn==0.12.0 plyfile ninja tqdm diffusers transformers accelerate timm einops matplotlib plotly typing argparse gradio kaleido==0.1.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install "git+https://github.com/ashawkey/diff-gaussian-rasterization.git"
```

- clone this repo:
```
git clone https://github.com/imlixinyang/director3d.git
cd director3d
```

- download the pre-trained model by:
```
wget https://huggingface.co/imlixinyang/director3d/resolve/main/model.ckpt?download=true -O model.ckpt
```

## 🧐 General Usage

You can generate 3D scenes with camera trajectories by running the following command:
``` bash
python inference.py --export_all --text "a delicious hamburger on a wooden table."
```

This will take about 5 minutes per sample on a single A100 GPU (or 7 minutes per sample on a single RTX 3090 GPU).
The results, including videos, images, cameras and 3DGS (``.splat``&``.ply``), can be found in ``./exps/tmp``.

## 💡 Code Overview

Core code of three key components of Director3D can be found in:

- Cinematographer - Trajectory Diffusion Transformer (Traj-DiT) 
```
system_traj_dit.py
```

- Decorator - Gaussian-driven Multi-view Latent Diffusion Model  (GM-LDM) 
```
system_gm_ldm.py
gm_ldm.py
```

- Detailer - SDS++
```
modules/refiners/sds_pp_refiner.py
```

<!-- ## 🚀 GUI Demo

Also, you can try out with GUI:

``` bash
python gradio_app.py
``` -->

<!-- ## Acknowledgement -->

## ❓ FAQ

1. ``torch.cuda.OutOfMemoryError: CUDA out of memory.`` 

  Please refer to [this issue](https://github.com/imlixinyang/Director3D/issues/4#issuecomment-2210183755)

 
## Citation

```
@article{li2024director3d,
  author = {Xinyang Li and Zhangyu Lai and Linning Xu and Yansong Qu and Liujuan Cao and Shengchuan Zhang and Bo Dai and Rongrong Ji},
  title = {Director3D: Real-world Camera Trajectory and 3D Scene Generation from Text},
  journal = {arXiv:2406.17601},
  year = {2024},
}
```


## License

Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)


The code is released for academic research use only. 

If you have any questions, please contact me via [imlixinyang@gmail.com](mailto:imlixinyang@gmail.com). 

