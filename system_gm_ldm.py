import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDIMScheduler
from gm_ldm import GaussianDrivenLDM
import tqdm

class GMLDMSystem(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.image_size = self.opt.network.image_size
        self.latent_size = self.opt.network.latent_size
        self.latent_channel = self.opt.network.latent_channel
        
        self.model = GaussianDrivenLDM(opt)

        self.scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type="sample", clip_sample=False, steps_offset=9, rescale_betas_zero_snr=True, set_alpha_to_one=True)

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)
     
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)
        
        self.num_input_views = self.opt.network.num_input_views

    def to(self, device):
        self.device = device
        return super().to(device)

    def inference_one_step(self, cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, guidance_scale=7.5, use_3d_mode=True):
        _latents_noisy = latents_noisy.clone()
        B, N, _, _ ,_ = latents_noisy.shape

        _t = t[..., None].repeat(1, N)

        uncond_latents_noisy = latents_noisy.clone()

        uncond_t = _t.clone()

        if use_3d_mode:

            latents_noisy = latents_noisy
            cameras = cameras
            text_embeddings = text_embeddings
            tt = _t

            B, N = latents_noisy.shape[:2]
            _, gaussians = self.model.denoise(latents_noisy, text_embeddings, tt, cameras)

            images_pred, _, _, _, _ = self.model.render(cameras, gaussians, h=self.image_size, w=self.image_size)

            _latents_pred = self.model.encode_image(images_pred)

            latents_less_noisy = self.scheduler.step(_latents_pred.cpu(), t.cpu(), _latents_noisy.cpu(), eta=1).prev_sample.to(self.device)
            
        else:  
            num_views = None
            cameras = torch.cat([cameras, cameras], 0)

            latents_noisy = torch.cat([latents_noisy, uncond_latents_noisy], 0)
            text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings], 0)
            tt = torch.cat([_t, uncond_t], 0)

            latents2d_pred = self.model.denoise(latents_noisy, text_embeddings, tt, cameras, return_3d=False, num_views=num_views)
            
            latents_pred, uncond_latents_pred = latents2d_pred.chunk(2, dim=0)
            _latents_pred = (latents_pred - uncond_latents_pred) * guidance_scale + uncond_latents_pred
        
            latents_less_noisy = self.scheduler.step(_latents_pred.cpu(), t.cpu(), _latents_noisy.cpu(), eta=0).prev_sample.to(self.device)

        if use_3d_mode:
            return latents_less_noisy, {"gaussians": gaussians, "images_pred": images_pred}
        else:
            return latents_less_noisy, {'latents_pred': _latents_pred}
    
    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def inference(self, cameras, text, dense_cameras=None, refiner=None, num_inference_steps=100, guidance_scale=7.5, use_3d_mode_every_m_steps=10, negative_text=""):
        B, N = cameras.shape[:2]
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        latents_noisy = torch.randn(B, N, self.latent_channel, self.latent_size, self.latent_size, device=self.device)

        text_embeddings = self.model.encode_text([text])
        uncond_text_embeddings =  self.model.encode_text([negative_text]).repeat(B, 1, 1)

        assert use_3d_mode_every_m_steps != 1, "use_3d_mode_every_m_steps can not be 1"

        if use_3d_mode_every_m_steps == -1:
            guidance_scale = guidance_scale
        else:
            guidance_scale = guidance_scale * use_3d_mode_every_m_steps / (use_3d_mode_every_m_steps - 1)

        for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps), desc='Denoising image sequence...'):
            if use_3d_mode_every_m_steps == -1:
                use_3d_mode = False
            else:
                use_3d_mode = (((len(timesteps) - 1 - i) % use_3d_mode_every_m_steps) == 0) 

            t = t[None].repeat(B)
            
            latents_noisy, result = self.inference_one_step(cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, guidance_scale=guidance_scale, use_3d_mode=use_3d_mode)

        if refiner is not None:
            assert 'gaussians' in result
            assert dense_cameras is not None
            gaussians = refiner.refine_gaussians(result['gaussians'], text, dense_cameras=dense_cameras)
            images_pred, _, _, _, _ = self.model.render(cameras, gaussians, h=self.image_size, w=self.image_size)
            result = {"gaussians": gaussians, "images_pred": images_pred}

        return result 