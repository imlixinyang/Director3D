import gradio as gr
from addict import Dict
import os
import torch
from system_gm_ldm import GMLDMSystem
from system_traj_dit import TrajDiTSystem
import numpy as np
from utils import export_ply_for_gaussians, import_str, export_video

import torch.nn.functional as F

import argparse
from omegaconf import OmegaConf
import functools
import gradio as gr

import base64
import io
import matplotlib
import matplotlib.pyplot as plt
import plotly
from pytorch3d.renderer import PerspectiveCameras
from typing import NamedTuple
from pytorch3d.vis.plotly_vis import plot_scene
import tempfile
import shutil
from contextlib import contextmanager
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--ckpt", default="model.ckpt")
    parser.add_argument("--gpu", type=int, default=0)
    return parser

def get_w2c(camera):
    w2c = np.linalg.inv(camera)
    return w2c

HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
<body><img src="data:image/png;charset=utf-8;base64, {image_encoded}"/>
{plotly_html}</body></html>"""

class AxisArgs(NamedTuple):  # pragma: no cover
    showgrid: bool = False
    zeroline: bool = False
    showline: bool = False
    ticks: str = ""
    showticklabels: bool = False
    backgroundcolor: str = "#fff"
    showaxeslabels: bool = False

@contextmanager
def temporary_directory():
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)
        
class App():
    def __init__(self):
        
        self.init_data()
        self.parser = get_args_parser()
    
        self.args, _ = self.parser.parse_known_args()

        self.opt = OmegaConf.load(self.args.config)

        self.device = f'cuda:{self.args.gpu}'
        self.ckpt = self.args.ckpt

        self.app = gr.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="Director3d Gradio Demo")
        with self.app:
            self.init_app()

    def init_data(self) :
        self.data = Dict(
        )

    def init_models(self):
        params = torch.load(self.ckpt, map_location=self.device)

        self.system_gm_ldm = GMLDMSystem(self.opt).to(self.device).eval()
        self.system_traj_dit = TrajDiTSystem(self.opt).to(self.device).eval()

        self.system_gm_ldm.model.load_state_dict(params['gm_ldm'], strict=False)
        self.system_traj_dit.model.load_state_dict(params['traj_dit'], strict=False)

        self.refiner = import_str(self.opt['inference']['refiner']['module'])(**self.opt['inference']['refiner']['args'], total_iterations=1000).to(self.device)

    def init_app(self):
        with temporary_directory() as tmp_dir:

            data = gr.State(
                    Dict({
                        "cameras": [],
                        "results": [],
                        "text": "",
                        "tmp_dir":tmp_dir
                    })
                )
            
            with gr.Row():

                helps = gr.Markdown(value=
                    """
                        # Gradio Demo of "🎥 Director3D: Real-world Camera Trajectory and 3D Scene Generation from Text"
                        
                        
                        <div>
                            <a style="display:inline-block" href="http://arxiv.org/abs/2404.11613"><img src='https://img.shields.io/badge/arXiv-Director3D-red?logo=arxiv' alt='Paper PDF'></a>
                            <a style="display:inline-block; margin-left: .5em" href='https://github.com/imlixinyang/director3d'><img src='https://img.shields.io/badge/Github-Director3D-blue?logo=github' alt='Project Page'></a>
                            <a style="display:inline-block; margin-left: .5em" href='https://imlixinyang.github.io/director3d-page'><img src='https://img.shields.io/badge/Project_Page-Director3D-green' alt='Project Page'></a>
                        </div>
                            
                        Abs: We introduce Director3D, a robust open-world text-to-3D generation framework, designed to generate both real-world 3D scenes and adaptive camera trajectories. To achieve this, (1) we first utilize a Trajectory Diffusion Transformer, acting as the <a style="background-color:#EFF185">Cinematographer</a>, to model the distribution of camera trajectories based on textual descriptions. Next, a Gaussian-driven Multi-view Latent Diffusion Model serves as the <a style="background-color:#93D881">Decorator</a>, modeling the image sequence distribution given the camera trajectories and texts. This model, fine-tuned from a 2D diffusion model, directly generates pixel-aligned 3D Gaussians as an immediate 3D scene representation for consistent denoising. Lastly, the 3D Gaussians are further refined by a novel SDS++ loss as the <a style="background-color:#6FE4E4">Detailer</a>, which incorporates the prior of the 2D diffusion model. 
                    """
                ) 
            
            with gr.Column():
                with gr.Row():
                    input_text = gr.Textbox(value="A delicious hamburger on a wooden table", label="Input Text",interactive=True)
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tab("Step 1: Cinematographer", interactive=True):
                        run_btn0 = gr.Button("Generate Camera Trajectories", variant='primary')
                        
                        # outcameras = gr.Gallery(label='Camera Trajectories', interactive=False, show_download_button=False, show_share_button=False)
                        
                        with gr.Column(scale=1):
                            with gr.Row():
                                outcamera1 = gr.Image(label='1', interactive=False, show_download_button=False, show_share_button=False)
                                outcamera2 = gr.Image(label='2', interactive=False, show_download_button=False, show_share_button=False)
                                
                        with gr.Column(scale=1):
                            with gr.Row():
                                outcamera3 = gr.Image(label='3', interactive=False, show_download_button=False, show_share_button=False)
                                outcamera4 = gr.Image(label='4', interactive=False, show_download_button=False, show_share_button=False)
                                
                        radio0 = gr.Radio(
                            [1, 2, 3, 4], label="Selected Trajectory", value=1, interactive=True, info="Which trajectory do you prefer for the next step?"
                        )
                    
                with gr.Column(scale=1):
                    with gr.Tab("Step 2: Decorator", interactive=True):
                        run_btn1 = gr.Button("Generate Coarse 3D Scenes", variant='primary')
                        
                        with gr.Column(scale=1):
                            with gr.Row():
                                outvideo1 = gr.Video(label='1', interactive=False, show_download_button=False, show_share_button=False, autoplay=True)
                                outvideo2 = gr.Video(label='2', interactive=False, show_download_button=False, show_share_button=False, autoplay=True)
                                
                        with gr.Column(scale=1):
                            with gr.Row():
                                outvideo3 = gr.Video(label='3', interactive=False, show_download_button=False, show_share_button=False, autoplay=True)
                                outvideo4 = gr.Video(label='4', interactive=False, show_download_button=False, show_share_button=False, autoplay=True)
                                
                        radio1 = gr.Radio(
                            [1, 2, 3, 4], label="Selected Scene", value=1, interactive=True, info="Which scene do you prefer for the next step?"
                        )
                    
                with gr.Column(scale=1):
                    with gr.Tab("Step 3: Detailer", interactive=True):
                        run_btn2 = gr.Button("Refine 3D Scene", variant='primary')
                        
                        outvideo_refined = gr.Video(label='Refined Scene', interactive=False,  show_download_button=False, show_share_button=False, autoplay=True)
                    
                run_btn0.click(fn= self.generate_camera_trajectory,
                            inputs=[input_text, data],
                            outputs=[outcamera1, outcamera2, outcamera3, outcamera4, data],
                            show_progress='full'
                            )
                
                run_btn1.click(fn= self.generate_coarse_3D_scenes,
                            inputs=[input_text, radio0, data],
                            outputs=[outvideo1, outvideo2, outvideo3, outvideo4, data],
                            show_progress='full'
                            )
                
                run_btn2.click(fn=self.refine_3D_scene,
                            inputs=[input_text, radio1, data],
                            outputs=[outvideo_refined, data],
                            show_progress='full'
                )   
        
    def plotly_scene_visualization(self, npy_file):
        camera_scene=[]
        cameras = np.load(npy_file)
        cameras = cameras[0]
        num_frames = cameras.shape[0]
        cameras = torch.from_numpy(cameras)
        
        camera = {}
        for i in range(0,num_frames,4):
            c2w = torch.eye(4)
            c2w[:3,:] =cameras[i][:12].reshape(3,4)
            fx,fy,cx,cy,H,W = cameras[i][12:].chunk(6,-1)
            K = [
                        [fx,   0,   cx,   0],
                        [0,   fy,   cy,   0],
                        [0,    0,    0,   1],
                        [0,    0,    1,   0],
                ]
            K = torch.tensor(K).unsqueeze(0)
            #fov = 2 * np.arctan2(H / 2, K[0, 0, 0]) 
            c2w[:3,1] *= -1
            c2w[:3,2] *= -1
            
            w2c = get_w2c(c2w)
            camera[i] = PerspectiveCameras(R=w2c[None,:3,:3], T=w2c[None,:3,3],K=K) 
                
        camera_scene.append(camera)
        # for visual convinience
        array_axs =[1, 0.5, 0,-0.5, -1] 
        array_axs = [x * 1.5 for x in array_axs]
        range_axs = [-1, 1]
        range_axs = [x * 1.5 for x in range_axs]
        dist = -2.7 
        elev = -52   
        azim = 180 

        # demo default view transform 
        R, T = look_at_view_transform(dist, elev, azim)

        # craeate view camera
        cameras_view = FoVPerspectiveCameras(R=R, T=T)
        fig = plot_scene(
            {" ":camera_scene[0]},
            yaxis={ "title": "",
                    "backgroundcolor":"rgb(200, 200, 230)",    
                    'tickmode': 'array',
                    'tickvals': array_axs,
                    'range':range_axs,
            },
            zaxis={ "title": "",
                    'tickmode': 'array',
                    'tickvals': array_axs,
                    'range':range_axs,
            },
            xaxis={ "title": "",
                    'tickmode': 'array',
                    'tickvals': array_axs,
                    'range':range_axs,
            },
            camera_scale=0.08,
            axis_args=AxisArgs(showline=False,showgrid=True,zeroline=False,showticklabels=False,showaxeslabels=False),
            viewpoint_cameras=cameras_view,
        )

        cmap = plt.get_cmap("hsv")
        for i in range(len(fig.data)):
            fig.data[i].line.color = matplotlib.colors.to_hex(cmap((i*4%32) / (num_frames)))
        fig.update_layout(showlegend=False)
        
        return fig

    def generate_camera_trajectory(self, text, data, progress=gr.Progress(track_tqdm=True)):
        data['cameras'] = []
        data['text'] = text
        # create cameras traj
        with torch.no_grad():
            video_paths= []
            for i in tqdm.tqdm(range(4)):
                print(i)
                filename = text
                extra_filename = f'_{i}'
                cameras = self.system_traj_dit.inference(data["text"])
                
                data['cameras'].append(cameras)    
                
                os.makedirs(os.path.join(data['tmp_dir'], 'camera'), exist_ok=True)
                np.save(os.path.join(data['tmp_dir'], 'camera', f'{filename}{extra_filename}.npy'), cameras.detach().cpu().numpy())
                
        # camearas traj visualization       
        tmp_dir = data['tmp_dir']
        image_out = f'{tmp_dir}/camera_visualization/images'
        html_out = f'{tmp_dir}/camera_visualization/htmls'
        os.makedirs(image_out, exist_ok=True)
        os.makedirs(html_out, exist_ok=True)
        
        directory = os.path.join(data['tmp_dir'], 'camera')
        files = os.listdir(directory)
        files = sorted(files)
        image_paths = []
        
        for file in files:
            if not file.endswith('.npy'):
                continue
            if not file.startswith(data["text"]):
                continue    
            file_path = os.path.join(directory, file)
                
            npy_files = file_path
            fig = self.plotly_scene_visualization(npy_files)
            image_path = f"{image_out}/{file[:-4]}.png"
            image_paths.append(image_path)
            plotly.io.write_image(fig, image_path, width=800, height=800)
            
            html_plot = plotly.io.to_html(fig, full_html=True, include_plotlyjs="cdn")
            s = io.BytesIO()
            plt.savefig(s, format="png", bbox_inches="tight")
            plt.close()
            image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            html_path_out = f"{html_out}/{file[:-4]}.html"
            with open(html_path_out, "w") as f:
                s = HTML_TEMPLATE.format(
                    image_encoded=image_encoded,
                    plotly_html=html_plot,
                )
                f.write(s)
        return image_paths[0], image_paths[1], image_paths[2], image_paths[3], data

    def generate_coarse_3D_scenes(self, text, camera_id, data, progress=gr.Progress(track_tqdm=True)):
        
        data['text'] = text
        data['results'] = []
        data['camera_id'] = camera_id
        
        cameras = data['cameras'][camera_id - 1]
          
        with torch.no_grad():
            
            video_paths = []
            
            for i in tqdm.tqdm(range(4)):
                filename = data["text"]
                
                extra_filename = f'_{i}'
                        
                sparse_cameras = cameras[:, ::int((cameras.shape[1]-1)/(self.system_gm_ldm.num_input_views-1))]
                result = self.system_gm_ldm.inference(sparse_cameras, data["text"], use_3d_mode_every_m_steps=10)
                
                data['results'].append(result)
                
                os.makedirs(os.path.join(data['tmp_dir'], 'video'), exist_ok=True)
                render_fn = lambda cameras, h=512, w=512: self.system_gm_ldm.model.render(cameras, result['gaussians'], h=h, w=w, bg_color=None)[:2]
                
                video_path = os.path.join(os.path.join(data['tmp_dir'], 'video'), f'{filename}{extra_filename}.mp4')
                
                export_video(render_fn, os.path.join(data['tmp_dir'], 'video'), f'{filename}{extra_filename}', cameras, device=self.device, fps=20, num_frames=240)
                
                video_paths.append(video_path)
                
        return video_paths[0], video_paths[1], video_paths[2], video_paths[3], data

    def refine_3D_scene(self, text, scene_id, data, progress=gr.Progress(track_tqdm=True)):
        
        data['text'] = text
        cameras = data['cameras'][data['camera_id'] - 1]  
        gaussians = data['results'][scene_id - 1]['gaussians']
        
        filename = text
        extra_filename = f'_{scene_id - 1}'
            
        gaussians = self.refiner.refine_gaussians(gaussians, text, dense_cameras=cameras)
            
        render_fn = lambda cameras, h=512, w=512: self.system_gm_ldm.model.render(cameras, gaussians, h=h, w=w, bg_color=None)[:2]
            
        video_path = os.path.join(os.path.join(data['tmp_dir'], 'video'), f'{filename}{extra_filename}_refined.mp4')
            
        export_video(render_fn, os.path.join(data['tmp_dir'], 'video'), f'{filename}{extra_filename}_refined', cameras, device=self.device)
            
        return video_path, data
        
    def launch(self, only_layout=False, **kwargs):
        if not only_layout:
            self.init_models()
        self.app.launch(**kwargs)
    
    def queue(self, **kwargs):
        self.app.queue(**kwargs)

if __name__ == "__main__":
    max_threads = 2
    api_open = True
    max_size = 10
    app = App()
    app.queue(api_open=api_open, max_size=max_size)
    app.launch(only_layout=False, share=False)
