import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from diffusers import StableDiffusionPipeline, DDIMScheduler

import tqdm
from utils import matrix_to_square
from modules.renderers.gaussians_renderer import quaternion_to_matrix, matrix_to_quaternion

from modules.dit import *

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c):
        shift_ca, scale_ca, gate_ca, shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_ca.unsqueeze(1) * self.cattn(modulate(self.norm1(x), shift_ca, scale_ca), y, y)
        x = x + gate_sa.unsqueeze(1) * self.sattn(modulate(self.norm2(x), shift_sa, scale_sa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class TrajDiTModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        pipe = StableDiffusionPipeline.from_pretrained(
            self.opt.network.sd_model_key, local_files_only=self.opt.network.local_files_only
        )
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False)
        
        del pipe

        hidden_size = opt.network.cdm.hidden_size
        num_blocks = opt.network.cdm.num_blocks
        num_tokens = opt.network.cdm.num_tokens

        self.t_embedder = nn.Sequential(
            TimestepEmbedder(hidden_size),
            nn.SiLU(),
        )

        self.y_embedder = nn.Linear(1024, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))

        self.blocks = nn.ModuleList([DiTBlock(hidden_size,**self.opt.network.cdm.block_args) for i in range(num_blocks)])

        self.in_block = nn.Linear(4 + 3 + 4, hidden_size)
        self.out_block = nn.Linear(hidden_size, 4 + 3 + 4)

    @torch.no_grad()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder[0].mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[0].mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_block.weight, 0)
        nn.init.constant_(self.out_block.bias, 0)
    
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
        return self.y_embedder(text_embeddings)

    def forward(self, x, y, t):

        x = self.in_block(x) + self.pos_embed

        t = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, y, t) 

        x = self.out_block(x)
        return x

class TrajDiTSystem(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.model = TrajDiTModel(opt)

        self.scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type="sample", clip_sample=False, steps_offset=9, rescale_betas_zero_snr=True, set_alpha_to_one=True)

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)
     
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)

    def to(self, device):
        self.device = device
        return super().to(device)
        
    def camera_to_token(self, cameras):
        B, N, _ = cameras.shape

        RT = cameras[:, :, :12].reshape(B, N, 3, 4)
        # rotation
        rotation = matrix_to_quaternion(RT[:, :, :, :3])
        # translation
        translation = RT[:, :, :, 3]
        # fx, fy, cx, cy
        intrinsics = torch.stack([cameras[:, :, 12] / cameras[:, :, 16], 
                                 cameras[:, :, 13] / cameras[:, :, 17], 
                                 cameras[:, :, 14] / cameras[:, :, 16], 
                                 cameras[:, :, 15] / cameras[:, :, 17]], dim=2)

        return torch.cat([rotation, translation, intrinsics], dim=2)

    def token_to_camera(self, tokens, image_size):
        B, N, _ = tokens.shape

        R = quaternion_to_matrix(tokens[:, :, :4]) # B, N, 3, 3
        T = tokens[:, :, 4:7].reshape(B, N, 3, 1) # B, N, 3, 1

        RT = torch.cat([R, T], dim=3).reshape(B, N, 12)

        intrinsics = torch.stack([tokens[:, :, 7] * image_size, 
                                  tokens[:, :, 8] * image_size, 
                                  tokens[:, :, 9] * image_size, 
                                  tokens[:, :, 10] * image_size,
                                  torch.full((B, N), fill_value=image_size, device=self.device),
                                  torch.full((B, N), fill_value=image_size, device=self.device),
                                 ], dim=2)

        return torch.cat([RT, intrinsics], dim=2)

    @torch.no_grad()
    def inference(self, text, num_inference_steps=100, image_size=512, return_each=False):
        B = 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        tokens_noisy = torch.randn(B, self.opt.network.cdm.num_tokens, 4 + 3 + 4, device=self.device)

        text_embeddings = self.model.encode_text([text])

        for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps), desc='Denoising camera trajectory...'):
            t = t[None].repeat(B)

            tokens_pred = self.model(tokens_noisy, text_embeddings, t)

            tokens_pred[:, :, :4] = F.normalize(tokens_pred[:, :, :4], dim=-1)
            
            tokens_noisy = self.scheduler.step(tokens_pred.cpu(), t.cpu(), tokens_noisy.cpu(), eta=0).prev_sample.to(self.device)

        return self.token_to_camera(tokens_noisy, image_size=image_size)