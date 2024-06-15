import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler

from modules.renderers.gaussians_renderer import GaussianRenderer

from utils import sample_from_dense_cameras, inverse_sigmoid

import tqdm

from .gs_utils import GaussiansManeger

class GSRefinerSDSPlusPlus(nn.Module):
    def __init__(self, 
            sd_model_key='stabilityai/stable-diffusion-2-1-base',
            local_files_only=True,
            num_views=1,
            total_iterations=500,
            guidance_scale=100,
            min_step_percent=0.02, 
            max_step_percent=0.75,
            lr_scale=1,
            lr_scale_end=1,
            lrs={'xyz': 0.0001, 'features': 0.01, 'opacity': 0.05, 'scales': 0.01, 'rotations': 0.01}, 
            use_lods=True,
            lambda_latent_sds=1,
            lambda_image_sds=0.01,
            lambda_image_variation=0,
            lambda_mask_variation=0, 
            lambda_mask_saturation=0,
            use_random_background_color=True,
            grad_clip=10,
            img_size=512,
            num_densifications=5,
            text_templete='$text$',
            negative_text_templete=''
        ):
        super().__init__()

        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_key, local_files_only=True
        )

        pipe.enable_xformers_memory_efficient_attention()
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False)
        self.vae = pipe.vae.requires_grad_(False)
        self.unet = pipe.unet.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            sd_model_key, subfolder="scheduler", local_files_only=True
        )
        
        del pipe

        self.num_views = num_views
        self.total_iterations = total_iterations
        self.guidance_scale = guidance_scale
        self.lrs = {key: value * lr_scale for key, value in lrs.items()}
        self.lr_scale = lr_scale
        self.lr_scale_end = lr_scale_end

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)

        self.device = 'cpu'

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.set_min_max_steps(min_step_percent, max_step_percent)

        self.renderer = GaussianRenderer()

        self.text_templete = text_templete
        self.negative_text_templete = negative_text_templete

        self.use_lods = use_lods

        self.lambda_latent_sds = lambda_latent_sds
        self.lambda_image_sds = lambda_image_sds

        self.lambda_image_variation = lambda_image_variation
        self.lambda_mask_variation = lambda_mask_variation

        self.lambda_mask_saturation = lambda_mask_saturation

        self.grad_clip = grad_clip
        self.img_size = img_size

        self.use_random_background_color = use_random_background_color

        self.opacity_threshold = 0.01
        self.densification_interval = self.total_iterations // (num_densifications + 1)

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    def to(self, device):
        self.device = device
        return super().to(device)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation_strategy='longest_first',
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(inputs.input_ids.to(next(self.text_encoder.parameters()).device))[0]
        return text_embeddings
    
    # @torch.cuda.amp.autocast(enabled=False)
    def encode_image(self, images):
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    # @torch.cuda.amp.autocast(enabled=False)
    def decode_latent(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample
        return images
    
    def train_step(
        self,
        images,
        t,
        text_embeddings,
        uncond_text_embeddings,
        learnable_text_embeddings,
    ):
        latents = self.encode_image(images)

        with torch.no_grad():
            B = latents.shape[0]
            t = t.repeat(self.num_views)
            
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

        if self.use_lods:
            with torch.enable_grad():
                noise_pred_learnable = self.unet(
                    latents_noisy, 
                    t, 
                    encoder_hidden_states=learnable_text_embeddings
                ).sample

            loss_embedding = F.mse_loss(noise_pred_learnable, noise, reduction="mean")
        else:
            noise_pred_learnable = noise
            loss_embedding = 0

        with torch.no_grad():
            noise_pred = self.unet(
                torch.cat([latents_noisy, latents_noisy], 0), 
                torch.cat([t, t], 0), 
                encoder_hidden_states=torch.cat([text_embeddings, uncond_text_embeddings], 0)
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)

            alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1) ** 0.5
            sigma = (1 - alpha) ** 0.5

            latents_pred = (latents_noisy - sigma * (noise_pred - noise_pred_learnable + noise)) / alpha
            images_pred = self.decode_latent(latents_pred).clamp(-1, 1)

        loss_latent_sds = (F.mse_loss(latents, latents_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / B
        loss_image_sds = (F.mse_loss(images, images_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / B

        return loss_latent_sds, loss_image_sds, loss_embedding

    @torch.cuda.amp.autocast(enabled=True)
    @torch.enable_grad()
    def refine_gaussians(self, gaussians, text, dense_cameras):
        
        gaussians_original = gaussians
        xyz, features, opacity, scales, rotations = gaussians

        mask = opacity[..., 0] >= self.opacity_threshold

        xyz_original = xyz[mask][None]
        features_original = features[mask][None]
        opacity_original = opacity[mask][None]
        scales_original = scales[mask][None]
        rotations_original = rotations[mask][None]

        text = self.text_templete.replace('$text$', text)

        text_embeddings = self.encode_text([text])
        uncond_text_embeddings =  self.encode_text([self.negative_text_templete.replace('$text$', text)])

        class LearnableTextEmbeddings(nn.Module):
            def __init__(self, uncond_text_embeddings):
                super().__init__()
                self.embeddings = nn.Parameter(torch.zeros_like(uncond_text_embeddings.float().detach().clone()))
                self.to(self.embeddings.device)

            def forward(self, cameras):
                B = cameras.shape[1]
                return self.embeddings.repeat(B, 1, 1)

        _learnable_text_embeddings = LearnableTextEmbeddings(uncond_text_embeddings)

        text_embeddings = text_embeddings.repeat(self.num_views, 1, 1)
        uncond_text_embeddings = uncond_text_embeddings.repeat(self.num_views, 1, 1)

        new_gaussians = GaussiansManeger(xyz_original, features_original, opacity_original, scales_original, rotations_original, self.lrs)

        optimizer_embeddings = torch.optim.Adam(_learnable_text_embeddings.parameters(), lr=self.lrs['embeddings'])

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(new_gaussians.optimizer, gamma=(self.lr_scale_end / self.lr_scale) ** (1 / self.total_iterations))

        for i in tqdm.trange(self.total_iterations, desc='Refining...'):

            if i % self.densification_interval == 0 and i != 0:
                new_gaussians.densify_and_prune()

            with torch.cuda.amp.autocast(enabled=False):   
                cameras = sample_from_dense_cameras(dense_cameras, torch.rand(1, self.num_views).to(self.device))

                learnable_text_embeddings = _learnable_text_embeddings(cameras)

                if self.lambda_mask_variation > 0 or self.lambda_image_variation > 0:
                    with torch.no_grad():
                        images_original, _, masks_original, _, _ = self.renderer(cameras, gaussians_original, bg_color='random', h=self.img_size, w=self.img_size)

                gaussians = new_gaussians()
                images_pred, depths_pred, masks_pred, reg_losses, _ = self.renderer(cameras, gaussians, bg_color='random', h=self.img_size, w=self.img_size)
            
            t = torch.full((1,), int((i / self.total_iterations) ** (1/2) * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
            # t = torch.randint(self.min_step, self.max_step, (self.num_views,), dtype=torch.long, device=self.device)
            loss_latent_sds, loss_img_sds, loss_embedding = self.train_step(images_pred.squeeze(0), t, text_embeddings, uncond_text_embeddings, learnable_text_embeddings)

            loss = loss_latent_sds * self.lambda_latent_sds + loss_img_sds * self.lambda_image_sds + loss_embedding

            if self.lambda_mask_variation > 0 or self.lambda_image_variation > 0:
                loss += self.lambda_mask_variation * F.mse_loss(masks_original, masks_pred, reduction='sum') / self.num_views
                loss += self.lambda_image_variation * F.mse_loss(images_original, images_pred, reduction='sum') / self.num_views

            if self.lambda_mask_saturation > 0:
                loss += self.lambda_mask_saturation * F.mse_loss(masks_pred, torch.ones_like(masks_pred), reduction='sum') / self.num_views

            # self.lambda_scale_regularization
            if True:
                scales = torch.exp(new_gaussians._scales)
                big_points_ws = scales.max(dim=1).values > 0.1
                loss += 10 * scales[big_points_ws].sum()
                
            loss.backward()

            new_gaussians.optimizer.step()
            new_gaussians.optimizer.zero_grad()

            optimizer_embeddings.step()
            optimizer_embeddings.zero_grad()

            lr_scheduler.step()
            
            for radii, viewspace_points in zip(self.renderer.radii, self.renderer.viewspace_points):
                visibility_filter = radii > 0
                new_gaussians.is_visible[visibility_filter] = 1
                new_gaussians.max_radii2D[visibility_filter] = torch.max(new_gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                new_gaussians.add_densification_stats(viewspace_points, visibility_filter)

        gaussians = new_gaussians()
        is_visible = new_gaussians.is_visible.bool()
        gaussians = [p[:, is_visible].detach() for p in gaussians]

        del new_gaussians
        return gaussians
