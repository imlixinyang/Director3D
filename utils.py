import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import imageio
from math import exp
import random
import kiui
import math
import tqdm

from io import BytesIO
import importlib

from modules.renderers.gaussians_renderer import quaternion_to_matrix, matrix_to_quaternion

from plyfile import PlyData, PlyElement

def import_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def matrix_to_square(mat):
    l = len(mat.shape)
    if l==3:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
    elif l==4:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)


@torch.no_grad()
def export_video(render_fn, save_path, name, dense_cameras, fps=60, num_frames=720, render_size=512, device='cuda:0'):

    images = []
    depths = []

    for i in tqdm.trange(num_frames, desc="Rendering video..."):

        t = torch.full((1, 1), fill_value=i/num_frames, device=device)

        camera = sample_from_dense_cameras(dense_cameras, t)
        
        image, depth = render_fn(camera, render_size, render_size)

        images.append(image.reshape(3, render_size, render_size).permute(1,2,0).detach().cpu().mul(1/2).add(1/2).clamp(0, 1).mul(255).numpy().astype(np.uint8))
        depths.append(depth.reshape(1, render_size, render_size).permute(1,2,0).detach().cpu().mul(1/2).add(1/2).clamp(0, 1).mul(255).numpy().astype(np.uint8))
        
    imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), images, fps=fps, quality=8, macro_block_size=1)
    
@torch.cuda.amp.autocast(enabled=False)
def quaternion_slerp(
    q0, q1, fraction, spin: int = 0, shortestpath: bool = True
):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = q1[d < 0.0]

    d = d.clamp(0, 1.0)

    # theta = torch.arccos(d) * fraction
    # q2 = q1 - q0 * d
    # q2 = q2 / (q2.norm(dim=-1) + 1e-10)
    
    # return torch.cos(theta) * q0 + torch.sin(theta) * q2

    angle = torch.acos(d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle)+ 1e-10)
    q0_ = q0 * torch.sin((1.0 - fraction) * angle) * isin
    q1_ = q1 * torch.sin(fraction * angle) * isin

    q = q0_ + q1_
    q[angle < 1e-5, :] = q0

    return q

def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths=[0, 0]):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """

    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])

    quaternion = quaternion_slerp(quat_a, quat_b, fraction)
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    R = quaternion_to_matrix(quaternion)
    T = (1 - fraction) * pose_a[..., :3, 3] + fraction * pose_b[..., :3, 3]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :3, :3] = R
    new_pose[..., :3, 3] = T
    return new_pose

def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0, 0, 0]):
    B, N, C = dense_cameras.shape
    B, M = t.shape
    
    left = torch.floor(t * (N-1)).long().clamp(0, N-2)
    right = left + 1
    fraction = t * (N-1) - left

    a = torch.gather(dense_cameras, 1, left[..., None].repeat(1, 1, C))
    b = torch.gather(dense_cameras, 1, right[..., None].repeat(1, 1, C))

    new_pose = sample_from_two_pose(a[:, :, :12].reshape(B, M, 3, 4), 
                                    b[:, :, :12].reshape(B, M, 3, 4), fraction, noise_strengths=noise_strengths[:2])

    new_ins = (1 - fraction) * a[:, :, 12:] + fraction * b[:, :, 12:]

    return torch.cat([new_pose.reshape(B, M, 12), new_ins], dim=2)


@torch.cuda.amp.autocast(enabled=False)
def sample_rays(cameras: torch.Tensor, gt=None, h=None, w=None, N=-1, P=None):
    ''' get rays
    Args:
        cameras: [B, 18], cam2world poses[12]&intrinsics[4]&HW[2]
        h, w, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = cameras.device
    B = cameras.shape[0]
    c2w = torch.eye(4)[None].to(device).repeat(B, 1, 1)
    c2w[:, :3, :] = cameras[:, :12].reshape(B, 3, 4)
    fx, fy, cx, cy, H, W = cameras[:, 12:].chunk(6, -1) # each
    
    if h is not None:
        fx, cx = fx * h / H, cx * h / H
    else:
        h = int(H[0].item())
    if w is not None:
        fy, cy = fy * w / W, cy * w / W
    else:
        w = int(W[0].item())

    if N > 0:
        if P is not None:   
            assert N % (P ** 2) == 0
            num_patch = N // (P ** 2)

            short_side = min(h, w)
            max_multiplier = short_side // P

            multiplier = torch.randint(1, max_multiplier + 1, size=[B * num_patch], device=device)
            offset_i =  (torch.rand(B * num_patch, device=device) * (h - P * (multiplier) + multiplier)).floor_().long()
            offset_j =  (torch.rand(B * num_patch, device=device) * (w - P * (multiplier) + multiplier)).floor_().long()

            i = torch.arange(0, P, device=device).expand(B * num_patch, P) * multiplier[..., None] + offset_i[..., None] 
            j = torch.arange(0, P, device=device).expand(B * num_patch, P) * multiplier[..., None] + offset_j[..., None] 

            inds = (i.reshape(B * num_patch, P, 1) * w + j.reshape(B * num_patch, 1, P)).reshape(B, -1)
            
        else:
            inds = torch.rand(B, N, device=device).mul(h*w).floor_().long() # may duplicate
    else:
        inds = torch.arange(0, h*w, device=device).expand(B, h*w)
        
    i = inds % w + 0.5
    j = torch.div(inds, w, rounding_mode='floor') + 0.5

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)

    # B x N x 3 & B x 3 x 3
    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)

    rays_o = c2w[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]
    
    if gt is not None:
        if gt.shape[2] != h or gt.shape[3] != w:
            gt = F.interpolate(gt, size=(h,w), align_corners=False, mode='bilinear')
        rays_gt = torch.gather(gt.reshape(B, 3, -1).permute(0, 2, 1), 1, torch.stack(3 * [inds], -1))
        return rays_o, rays_d, rays_gt

    return rays_o, rays_d

def get_camera2world(elev_rad, azim_rad, roll, radius):
    R_elev = np.array([[np.cos(elev_rad), 0, np.sin(elev_rad)],
                           [0, 1, 0],
                           [-np.sin(elev_rad), 0, np.cos(elev_rad)]])

    R_azim = np.array([[np.cos(azim_rad), -np.sin(azim_rad), 0],
                           [np.sin(azim_rad), np.cos(azim_rad), 0],
                           [0, 0, 1]])

    R_up = np.array([[np.cos(np.radians(90 + roll)), -np.sin(np.radians(90 + roll)), 0],
                         [np.sin(np.radians(90 + roll)), np.cos(np.radians(90 + roll)), 0],
                         [0, 0, 1]])

    R = np.dot(R_elev, R_azim)
    R = np.dot(R_up, R)

    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = radius

    rot = np.eye(4, dtype=np.float32)
    rot[:3, :3] = R.T
        
    c2w = rot @ c2w
    return c2w
    
def get_random_cameras(num_views=4, elev_range=[60, 120], azim_range=[0, 360], dist_range=[1.7, 2.0], focal_range=[560, 560], intrinsic=[512/2, 512/2, 512, 512], determined=False, ref_camera=None, inplace_first=False):
    
    c2ws = []
    focals = []

    for i in range(num_views):
        if determined:
            azim = i / num_views * (azim_range[1] - azim_range[0]) + azim_range[0]
            elev = (elev_range[0] + elev_range[1]) / 2
            dist = (dist_range[0] + dist_range[1]) / 2
            focal = (focal_range[0] + focal_range[1]) / 2
        else:
            azim, elev, dist, focal = ((random.random() * (r[1] - r[0]) + r[0]) for r in (azim_range, elev_range, dist_range, focal_range))

        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)

        c2w = get_camera2world(elev_rad, azim_rad, 0, dist)

        c2ws.append(c2w)
        focals.append(focal)
    
    c2ws = torch.from_numpy(np.stack(c2ws, axis=0)).float()
    focals = torch.from_numpy(np.stack(focals, axis=0)).float().unsqueeze(1)

    if ref_camera is not None:
        ref_c2w = matrix_to_square(ref_camera[..., :12].reshape(1, 3, 4)).detach().cpu()
        ref_w2c = torch.inverse(ref_c2w)
        c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws)

        if inplace_first:
            c2ws[0] = torch.eye(4)

    cameras = torch.cat([c2ws[:,:3,:].flatten(1, 2), focals, focals, torch.Tensor([*intrinsic])[None].repeat(num_views, 1)], dim=1)[None]

    return cameras

def export_ply_for_gaussians(path, gaussians, opacity_threshold=0.00, bbox=[-2, 2], compatible=True):

    xyz, features, opacity, scales, rotations = gaussians

    # assert xyz.shape[0] == 1
     
    means3D = xyz[0].contiguous().float()
    opacity = opacity[0].contiguous().float()
    scales = scales[0].contiguous().float()
    rotations = rotations[0].contiguous().float()
    shs = features[0].contiguous().float() # [N, 1, 3]

    # prune by opacity
    mask = opacity[..., 0] >= opacity_threshold
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(path + '.ply')

    plydata = PlyData([el])

    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    with open(path + '.splat', "wb") as f:
        f.write(buffer.getvalue())


def load_ply_for_gaussians(path, device='cpu', compatible=True):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    print("Number of points at loading : ", xyz.shape[0])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float, device=device)[None]
    features = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2)[None]
    opacity = torch.tensor(opacities, dtype=torch.float, device=device)[None]
    scales = torch.tensor(scales, dtype=torch.float, device=device)[None]
    rotations = torch.tensor(rots, dtype=torch.float, device=device)[None]

    if compatible:
        opacity = torch.sigmoid(opacity)
        scales = torch.exp(scales)

    return xyz, features, opacity, scales, rotations
