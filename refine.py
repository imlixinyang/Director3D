import os
import torch
import random
import numpy as np
import imageio
import tqdm
from utils import load_ply_for_gaussians, export_ply_for_gaussians, import_str, export_video

from PIL import Image
import torch.nn.functional as F

from torchvision.utils import save_image

if __name__ == "__main__":
    
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--ply", type=str)
    parser.add_argument("--camera", type=str)

    parser.add_argument("--text", default="a delicious hamburger on a wooden table")
    
    parser.add_argument("--text_templete", default="$text$")

    parser.add_argument("--export_all", action='store_true', default=False)
    parser.add_argument("--export_video", action='store_true', default=False)
    parser.add_argument("--export_ply", action='store_true', default=False)

    parser.add_argument("--num_refine_steps", type=int, default=1000)
    parser.add_argument("--out_dir", default='./exps/tmp')
    parser.add_argument("--gpu", type=int, default=0)
    args, extras = parser.parse_known_args()

    print(args)
    
    args.export_video = args.export_video or args.export_all
    args.export_ply = args.export_ply or args.export_all

    opt = OmegaConf.load(args.config)

    torch.backends.cudnn.benchmark = True

    device = f'cuda:{args.gpu}'
    
    if args.num_refine_steps > 0:
        refiner = import_str(opt['inference']['refiner']['module'])(**opt['inference']['refiner']['args'], total_iterations=args.num_refine_steps).to(device)

    text = args.text

    gaussians = load_ply_for_gaussians(args.ply, device=device)
    cameras = torch.tensor(np.load(args.camera), dtype=torch.float, device=device)

    with torch.no_grad():

        if True:

            filename = text 
            text = args.text_templete.replace('$text$', text)

            if args.num_refine_steps > 0:
                refined_gaussians = refiner.refine_gaussians(gaussians, text, dense_cameras=cameras)                        

            if args.export_ply:
                os.makedirs(os.path.join(args.out_dir, 'ply'), exist_ok=True)
                export_ply_for_gaussians(os.path.join(args.out_dir, 'ply', f'{filename}_refined'), refined_gaussians)
                    
            if args.export_video:
                os.makedirs(os.path.join(args.out_dir, 'video'), exist_ok=True)
                render_fn = lambda cameras, h=512, w=512: refiner.renderer(cameras, refined_gaussians, h=h, w=w, bg_color=None)[:2]
                export_video(render_fn, os.path.join(args.out_dir, 'video') , f'{filename}_refined', cameras, device=device)