import torch    
import torch.nn as nn
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_softplus(x, beta=1):
    return (torch.exp(beta * x) - 1).log() / beta

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


class GaussiansManeger():
    def __init__(self, xyz, features, opacity, scales, rotations, lrs):
        self._xyz = nn.Parameter(xyz.squeeze(0).contiguous().float().detach().clone().requires_grad_(True))
        self._features = nn.Parameter(features.squeeze(0).contiguous().float().detach().clone().requires_grad_(True))
        self._opacity = nn.Parameter(inverse_sigmoid(opacity.squeeze(0)).contiguous().float().detach().clone().requires_grad_(True))
        self._scales = nn.Parameter(torch.log(scales.squeeze(0) + 1e-8).contiguous().float().detach().clone().requires_grad_(True))
        self._rotations = nn.Parameter(rotations.squeeze(0).contiguous().float().detach().clone().requires_grad_(True))

        self.device = self._xyz.device

        self.optimizer = torch.optim.Adam([{"name": "xyz", 'params': [self._xyz], 'lr': lrs['xyz']},
                                            {"name": "features", 'params': [self._features], 'lr': lrs['features']},
                                            {"name": "opacity", 'params': [self._opacity], 'lr': lrs['opacity']},
                                            {"name": "scales", 'params': [self._scales], 'lr': lrs['scales']},
                                            {"name": "rotations", 'params': [self._rotations], 'lr': lrs['rotations']},
                                            ], betas=(0.9, 0.99), lr=0.0, eps=1e-15)

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self._xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device=self.device)
        self.is_visible = torch.zeros((self._xyz.shape[0],), device=self.device)

        self.percent_dense = 0.003

    def __call__(self):
        xyz = self._xyz 
        features = self._features
        opacity = torch.sigmoid(self._opacity)
        scales = torch.exp(self._scales)
        rotations = torch.nn.functional.normalize(self._rotations, dim=-1)
        return xyz.unsqueeze(0), features.unsqueeze(0), opacity.unsqueeze(0), scales.unsqueeze(0), rotations.unsqueeze(0)

    @torch.no_grad()
    def densify_and_prune(self, max_grad=4, extent=2, opacity_threshold=0.001, K=1, max_num_points=500000, max_screen_size=2):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if True:
            self.densify_and_clone(grads, max_grad, scene_extent=extent)
            self.densify_and_split(grads, max_grad, scene_extent=extent)

        prune_mask = (torch.sigmoid(self._opacity) < opacity_threshold).squeeze() # or torch.exp(self._scales.max(dim=1).values) < 0.001

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def prune_points(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features = optimizable_tensors["features"]
        self._opacity = optimizable_tensors["opacity"]
        self._scales = optimizable_tensors["scales"]
        self._rotations = optimizable_tensors["rotations"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.is_visible = self.is_visible[valid_points_mask]

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def add_points(self, params):

        num_points = params["xyz"].shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._xyz = optimizable_tensors["xyz"]
        self._features = optimizable_tensors["features"]
        self._opacity = optimizable_tensors["opacity"]
        self._scales = optimizable_tensors["scales"]
        self._rotations = optimizable_tensors["rotations"]

        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_points, 1), device=self.device)])
        self.denom = torch.cat([self.denom, torch.zeros((num_points, 1), device=self.device)])
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((num_points,), device=self.device)])
        self.is_visible = torch.cat([self.is_visible, torch.zeros((num_points,), device=self.device)])

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold=0.04, scene_extent=2, N=2):
        n_init_points = self._xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(torch.exp(self._scales), dim=1).values > self.percent_dense * scene_extent
        )

        stds = torch.exp(self._scales[selected_pts_mask]).repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(self._rotations[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scales = torch.log(torch.exp(self._scales[selected_pts_mask]) / (0.8*N)).repeat(N, 1)
        new_rotations = self._rotations[selected_pts_mask].repeat(N, 1)
        new_features = self._features[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        params = {  "xyz": new_xyz,
                    "features": new_features,
                    "opacity": new_opacity,
                    "scales" : new_scales,
                    "rotations" : new_rotations }
        
        self.add_points(params)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold=0.02, scene_extent=2):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(torch.exp(self._scales), dim=1).values <= self.percent_dense * scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features = self._features[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scales = self._scales[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]

        params = {  "xyz": new_xyz,
                    "features": new_features,
                    "opacity": new_opacity,
                    "scales" : new_scales,
                    "rotations" : new_rotations }

        self.add_points(params)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1