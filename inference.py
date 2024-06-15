import os
import torch
import random
from system_gm_ldm import GMLDMSystem
from system_traj_dit import TrajDiTSystem
import numpy as np
import imageio
import tqdm
from utils import sample_from_dense_cameras, export_ply_for_gaussians, import_str, matrix_to_square, export_video

from PIL import Image
import torch.nn.functional as F

from torchvision.utils import save_image

if __name__ == "__main__":
    
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--ckpt", default="model.ckpt")

    parser.add_argument("--text_file", default=None)

    parser.add_argument("--text", default="a delicious hamburger on a wooden table.")
    
    parser.add_argument("--text_templete", default="$text$")

    parser.add_argument("--export_all", action='store_true', default=False)
    parser.add_argument("--export_video", action='store_true', default=False)
    parser.add_argument("--export_camera", action='store_true', default=False)
    parser.add_argument("--export_ply", action='store_true', default=False)
    parser.add_argument("--export_image", action='store_true', default=False)

    parser.add_argument("--num_refine_steps", type=int, default=1000)
    parser.add_argument("--out_dir", default='./exps/tmp')
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--use_3d_mode_every_m_steps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    args, extras = parser.parse_known_args()

    print(args)

    args.export_video = args.export_video or args.export_all
    args.export_camera = args.export_camera or args.export_all
    args.export_ply = args.export_ply or args.export_all
    args.export_image = args.export_image or args.export_all

    opt = OmegaConf.load(args.config)

    torch.backends.cudnn.benchmark = True

    device = f'cuda:{args.gpu}'
    ckpt = args.ckpt

    params = torch.load(ckpt, map_location=device)

    system_gm_ldm = GMLDMSystem(opt).to(device).eval()
    system_traj_dit = TrajDiTSystem(opt).to(device).eval()

    system_gm_ldm.model.load_state_dict(params['gm_ldm'], strict=False)
    system_traj_dit.model.load_state_dict(params['traj_dit'], strict=False)

    refiner = import_str(opt['inference']['refiner']['module'])(**opt['inference']['refiner']['args'], total_iterations=args.num_refine_steps).to(device) if args.num_refine_steps > 0 else None

    if args.text_file is not None:
        with open(args.text_file, 'r') as f:
            texts = f.readlines()
            texts = [text.replace('\n', '') for text in texts]
    else:
        texts = [args.text]

    with torch.no_grad():

        for index, text in enumerate(texts):

            filename = text 
            print(text)

            text = args.text_templete.replace('$text$', text)

            for i in range(args.num_samples):
                print(i)

                extra_filename = '' if args.num_samples == 1 else f'_{i}'

                cameras = system_traj_dit.inference(text)
                
                sparse_cameras = cameras[:, ::int((cameras.shape[1]-1)/(system_gm_ldm.num_input_views-1))]

                result = system_gm_ldm.inference(sparse_cameras, text, dense_cameras=cameras, use_3d_mode_every_m_steps=args.use_3d_mode_every_m_steps, refiner=refiner)

                if args.export_image:
                    os.makedirs(os.path.join(args.out_dir, 'image'), exist_ok=True)
                    save_image(((result['images_pred'][0].permute(1, 2, 0, 3).reshape(3, system_gm_ldm.image_size, -1)+1)/2).clamp(0, 1).detach().cpu(), os.path.join(args.out_dir, 'image', f'{filename}{extra_filename}.png'))

                if args.export_camera:
                    os.makedirs(os.path.join(args.out_dir, 'camera'), exist_ok=True)
                    np.save(os.path.join(args.out_dir, 'camera', f'{filename}{extra_filename}.npy'), cameras.detach().cpu().numpy())

                if args.export_ply:
                    os.makedirs(os.path.join(args.out_dir, 'ply'), exist_ok=True)
                    export_ply_for_gaussians(os.path.join(args.out_dir, 'ply', f'{filename}{extra_filename}'), result['gaussians'])
                    
                if args.export_video:
                    os.makedirs(os.path.join(args.out_dir, 'video'), exist_ok=True)
                    render_fn = lambda cameras, h=512, w=512: system_gm_ldm.model.render(cameras, result['gaussians'], h=h, w=w, bg_color=None)[:2]
                    export_video(render_fn, os.path.join(args.out_dir, 'video') , f'{filename}{extra_filename}', cameras, device=device)

            
