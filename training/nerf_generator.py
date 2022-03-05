import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi
from scipy.spatial.transform import Rotation as Rot

from .nerf_util import (arange_pixels, get_camera_mat, get_camera_pose,
                        get_random_pose, image_points_to_world,
                        origin_to_world)


class Generator(nn.Module):
    """GIRAFFE Generator Class. without neural render.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        z_dim_bg (int): dimension of background latent code z_bg
        decoder (nn.Module): decoder network
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        background_generator (nn.Module): background generator
        bounding_box_generaor (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        fov (float): field of view
        background_rotation_range (tuple): background rotation range
         (0 - 1)
        sample_object-existance (bool): whether to sample the existence
            of objects; only used for clevr2345
        use_max_composition (bool): whether to use the max
            composition operator instead
    """
    def __init__(
            self,
            device,
            z_dim=256,
            z_dim_bg=128,
            decoder=None,
            range_u=(0, 0),
            range_v=(0.25, 0.25),
            n_ray_samples=64,
            range_radius=(2.732, 2.732),
            depth_range=[0.5, 6.],
            background_generator=None,
            bounding_box_generator=None,
            resolution_vol=36,  # change to fit s3
            fov=49.13,
            backround_rotation_range=[0., 0.],
            sample_object_existance=False,
            use_max_composition=False,
            **kwargs):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.bounding_box_generator = bounding_box_generator
        self.fov = fov
        self.backround_rotation_range = backround_rotation_range
        self.sample_object_existance = sample_object_existance
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.use_max_composition = use_max_composition

        self.camera_matrix = get_camera_mat(fov=fov).to(device)

        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        if background_generator is not None:
            self.background_generator = background_generator.to(device)
        else:
            self.background_generator = None
        if bounding_box_generator is not None:
            self.bounding_box_generator = bounding_box_generator.to(device)
        else:
            self.bounding_box_generator = bounding_box_generator

    def forward(self,
                batch_size=32,
                latent_codes=None,
                camera_matrices=None,
                transformations=None,
                bg_rotation=None,
                mode='training',
                it=0,
                return_alpha_map=False,
                not_render_background=False,
                only_render_background=False):
        # shape_obj, arr_obj, shape_bg, app_bg
        # [(bz, N, z_dim), (bz, N, z_dim), (bz, bg_dim), (bz, bg_dim)]
        # import ipdb
        # ipdb.set_trace()
        if latent_codes is None:
            latent_codes = self.get_latent_codes(batch_size)

        # [camera_mat, world2carmera_mat]
        if camera_matrices is None:
            camera_matrices = self.get_random_camera(batch_size)

        # [s, t, R] --> [scale, transformation, rotation]
        # [(bz, N, 3), (bz, N, 3), (bz, N, 3, 3)]
        if transformations is None:
            transformations = self.get_random_transformations(batch_size)

        # (bz, 3, 3)
        if bg_rotation is None:
            bg_rotation = self.get_random_bg_rotation(batch_size)

        if return_alpha_map:
            rgb_v, alpha_map = self.volume_render_image(
                latent_codes,
                camera_matrices,
                transformations,
                bg_rotation,
                mode=mode,
                it=it,
                return_alpha_map=True,
                not_render_background=not_render_background)
            return alpha_map
        else:
            rgb_v = self.volume_render_image(
                latent_codes,
                camera_matrices,
                transformations,
                bg_rotation,
                mode=mode,
                it=it,
                not_render_background=not_render_background,
                only_render_background=only_render_background)
            return rgb_v

    def get_n_boxes(self):
        if self.bounding_box_generator is not None:
            n_boxes = self.bounding_box_generator.n_boxes
        else:
            n_boxes = 1
        return n_boxes

    def get_latent_codes(self, batch_size=32, tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg

        n_boxes = self.get_n_boxes()

        def sample_z(x):
            return self.sample_z(x, tmp=tmp)

        z_shape_obj = sample_z((batch_size, n_boxes, z_dim))
        z_app_obj = sample_z((batch_size, n_boxes, z_dim))
        z_shape_bg = sample_z((batch_size, z_dim_bg))
        z_app_bg = sample_z((batch_size, z_dim_bg))

        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z

    def get_vis_dict(self, batch_size=32):
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': self.get_latent_codes(batch_size),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
            'bg_rotation': self.get_random_bg_rotation(batch_size)
        }
        return vis_dict

    def get_random_camera(self, batch_size=32, to_device=True):
        """Get random camera and matrixs.

        camera_mat: camera to screen
        world_mat: world to camera
        """
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_random_pose(self.range_u, self.range_v,
                                    self.range_radius, batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_camera(self,
                   val_u=0.5,
                   val_v=0.5,
                   val_r=0.5,
                   batch_size=32,
                   to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_camera_pose(self.range_u,
                                    self.range_v,
                                    self.range_radius,
                                    val_u,
                                    val_v,
                                    val_r,
                                    batch_size=batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_random_bg_rotation(self, batch_size, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(
                    Rot.from_euler('z', r_random * 2 * np.pi).as_dcm())
                for i in range(batch_size)
            ]
            R_bg = torch.stack(R_bg, dim=0).reshape(batch_size, 3, 3).float()
        else:
            # no rotation
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            R_bg = R_bg.to(self.device)
        return R_bg

    def get_bg_rotation(self, val, batch_size=32, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
            r = torch.from_numpy(
                Rot.from_euler('z', r_val * 2 * np.pi).as_dcm()).reshape(
                    1, 3, 3).repeat(batch_size, 1, 1).float()
        else:
            r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            r = r.to(self.device)
        return r

    def get_random_transformations(self, batch_size=32, to_device=True):
        device = self.device
        s, t, R = self.bounding_box_generator(batch_size)
        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations(self,
                            val_s=[[0.5, 0.5, 0.5]],
                            val_t=[[0.5, 0.5, 0.5]],
                            val_r=[0.5],
                            batch_size=32,
                            to_device=True):
        device = self.device
        s = self.bounding_box_generator.get_scale(batch_size=batch_size,
                                                  val=val_s)
        t = self.bounding_box_generator.get_translation(batch_size=batch_size,
                                                        val=val_t)
        R = self.bounding_box_generator.get_rotation(batch_size=batch_size,
                                                     val=val_r)

        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations_in_range(self,
                                     range_s=[0., 1.],
                                     range_t=[0., 1.],
                                     range_r=[0., 1.],
                                     n_boxes=1,
                                     batch_size=32,
                                     to_device=True):
        s, t, R = [], [], []

        def rand_s():
            return range_s[0] + \
                np.random.rand() * (range_s[1] - range_s[0])

        def rand_t():
            return range_t[0] + \
                np.random.rand() * (range_t[1] - range_t[0])

        def rand_r():
            return range_r[0] + \
                np.random.rand() * (range_r[1] - range_r[0])

        for i in range(batch_size):
            val_s = [[rand_s(), rand_s(), rand_s()] for j in range(n_boxes)]
            val_t = [[rand_t(), rand_t(), rand_t()] for j in range(n_boxes)]
            val_r = [rand_r() for j in range(n_boxes)]
            si, ti, Ri = self.get_transformations(val_s,
                                                  val_t,
                                                  val_r,
                                                  batch_size=1,
                                                  to_device=to_device)
            s.append(si)
            t.append(ti)
            R.append(Ri)
        s, t, R = torch.cat(s), torch.cat(t), torch.cat(R)
        if to_device:
            device = self.device
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_rotation(self, val_r, batch_size=32, to_device=True):
        device = self.device
        R = self.bounding_box_generator.get_rotation(batch_size=batch_size,
                                                     val=val_r)

        if to_device:
            R = R.to(device)
        return R

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self,
                                p,
                                transformations,
                                box_idx=0,
                                scale_factor=1.):
        r""" k^{-1}(x) in Eq. 7.
            k(x) = R S X + t (Eq. 6)
            k^{-1}(x) = ???
        """
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (
            p - bb_t[:, box_idx].unsqueeze(1)).permute(0, 2, 1)).permute(
                0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box

    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 rotation_matrix):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        camera_world = (
            rotation_matrix @ camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        pixels_world = (
            rotation_matrix @ pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
        ray_world = pixels_world - camera_world

        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_world.unsqueeze(-2).contiguous()
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert (p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations, i):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        # import ipdb
        # ipdb.set_trace()
        pixels_world_i = self.transform_points_to_box(pixels_world,
                                                      transformations, i)
        camera_world_i = self.transform_points_to_box(camera_world,
                                                      transformations, i)
        ray_i = pixels_world_i - camera_world_i

        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert (p_i.shape == ray_i.shape)

        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind,
                                     torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(1, -1, 1),
                                     torch.arange(ns).reshape(1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.ones_like(z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1. - torch.exp(-F.relu(sigma) * dists)
        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def get_object_existance(self, n_boxes, batch_size=32):
        '''
        Note: We only use this setting for Clevr2345, so that we can hard-code
        the probabilties here. If you want to apply it to a different scenario,
        you would need to change these.
        '''
        probs = [
            .19456788355146545395,
            .24355003312266127155,
            .25269546846185522711,
            .30918661486401804737,
        ]

        n_objects_prob = np.random.rand(batch_size)
        n_objects = np.zeros_like(n_objects_prob).astype(np.int)
        p_cum = 0
        obj_n = [i for i in range(2, n_boxes + 1)]
        for idx_p in range(len(probs)):
            n_objects[(n_objects_prob >= p_cum)
                      & (n_objects_prob < p_cum + probs[idx_p])] = obj_n[idx_p]
            p_cum = p_cum + probs[idx_p]
            assert (p_cum <= 1.)

        object_existance = np.zeros((batch_size, n_boxes))
        for b_idx in range(batch_size):
            n_obj = n_objects[b_idx]
            if n_obj > 0:
                idx_true = np.random.choice(n_boxes,
                                            size=(n_obj, ),
                                            replace=False)
                object_existance[b_idx, idx_true] = True
        object_existance = object_existance.astype(np.bool)
        return object_existance

    def volume_render_image(self,
                            latent_codes,
                            camera_matrices,
                            transformations,
                            bg_rotation,
                            mode='training',
                            it=0,
                            return_alpha_map=False,
                            not_render_background=False,
                            only_render_background=False):
        res = self.resolution_vol
        device = self.device
        n_steps = self.n_ray_samples
        n_points = res * res
        depth_range = self.depth_range
        batch_size = latent_codes[0].shape[0]
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = latent_codes
        assert (not (not_render_background and only_render_background))

        # Arange Pixels
        # use [1] get scales_pixels
        pixels = arange_pixels((res, res), batch_size,
                               invert_y_axis=False)[1].to(device)
        # TODO: why we do this
        #   --> equals to set invert_y_axis=True in arange_pixels
        pixels[..., -1] *= -1.
        # Project to 3D world
        # camera_matrices: [cam_mat, world_mat]
        pixels_world = image_points_to_world(pixels,
                                             camera_mat=camera_matrices[0],
                                             world_mat=camera_matrices[1])
        camera_world = origin_to_world(n_points,
                                       camera_mat=camera_matrices[0],
                                       world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
            torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        if mode == 'training':
            di = self.add_noise_to_interval(di)

        n_boxes = latent_codes[0].shape[1]
        feat, sigma = [], []
        n_iter = n_boxes if not_render_background else n_boxes + 1
        if only_render_background:
            n_iter = 1
            n_boxes = 0
        # import ipdb
        for i in range(n_iter):
            if i < n_boxes:  # Object
                # transformatios: [(bz, N, 3), (bz, N, 3), (bz, N, 3, 3)]
                p_i, r_i = self.get_evaluation_points(pixels_world,
                                                      camera_world, di,
                                                      transformations, i)
                z_shape_i, z_app_i = z_shape_obj[:, i], z_app_obj[:, i]

                feat_i, sigma_i = self.decoder(p_i, r_i, z_shape_i, z_app_i)

                if mode == 'training':
                    # As done in NeRF, add noise during training
                    sigma_i += torch.randn_like(sigma_i)

                # Mask out values outside
                padd = 0.1
                mask_box = torch.all(p_i <= 1. + padd, dim=-1) & torch.all(
                    p_i >= -1. - padd, dim=-1)
                sigma_i[mask_box == 0] = 0.

                # Reshape
                sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
                feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)
            else:  # Background
                p_bg, r_bg = self.get_evaluation_points_bg(
                    pixels_world, camera_world, di, bg_rotation)

                # ipdb.set_trace()
                feat_i, sigma_i = self.background_generator(
                    p_bg, r_bg, z_shape_bg, z_app_bg)
                sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
                feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)

                if mode == 'training':
                    # As done in NeRF, add noise during training
                    sigma_i += torch.randn_like(sigma_i)

            feat.append(feat_i)
            sigma.append(sigma_i)
        # ipdb.set_trace()
        sigma = F.relu(torch.stack(sigma, dim=0))
        feat = torch.stack(feat, dim=0)

        if self.sample_object_existance:
            object_existance = self.get_object_existance(n_boxes, batch_size)
            # add ones for bg
            object_existance = np.concatenate(
                [object_existance,
                 np.ones_like(object_existance[..., :1])],
                axis=-1)
            object_existance = object_existance.transpose(1, 0)
            sigma_shape = sigma.shape
            sigma = sigma.reshape(sigma_shape[0] * sigma_shape[1], -1)
            object_existance = torch.from_numpy(object_existance).reshape(-1)
            # set alpha to 0 for respective objects
            sigma[object_existance == 0] = 0.
            sigma = sigma.reshape(*sigma_shape)

        # Composite
        sigma_sum, feat_weighted = self.composite_function(sigma, feat)

        # Get Volume Weights
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)

        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, res,
                                                     res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        if return_alpha_map:
            n_maps = sigma.shape[0]
            acc_maps = []
            for i in range(n_maps - 1):
                sigma_obj_sum = torch.sum(sigma[i:i + 1], dim=0)
                weights_obj = self.calc_volume_weights(di,
                                                       ray_vector,
                                                       sigma_obj_sum,
                                                       last_dist=0.)
                acc_map = torch.sum(weights_obj, dim=-1, keepdim=True)
                acc_map = acc_map.permute(0, 2,
                                          1).reshape(batch_size, -1, res, res)
                acc_map = acc_map.permute(0, 1, 3, 2)
                acc_maps.append(acc_map)
            acc_map = torch.cat(acc_maps, dim=1)
            return feat_map, acc_map
        else:
            return feat_map


class Decoder(nn.Module):
    """Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    """
    def __init__(self,
                 hidden_size=128,
                 n_blocks=8,
                 n_blocks_view=1,
                 skips=[4],
                 use_viewdirs=True,
                 n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_dim=64,
                 rgb_out_dim=128,
                 final_sigmoid_activation=False,
                 downscale_p_by=2.,
                 positional_encoding='normal',
                 gauss_dim_pos=10,
                 gauss_dim_view=4,
                 gauss_std=4.,
                 **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        assert (positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1, gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1, gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        else:
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = nn.Linear(dim_embed, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)])
            self.fc_p_skips = nn.ModuleList(
                [nn.Linear(dim_embed, hidden_size) for i in range(n_skips)])
        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList([
                nn.Linear(dim_embed_view + hidden_size, hidden_size)
                for i in range(n_blocks_view - 1)
            ])

    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed),
                 torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc

            # ---> old code
            # p_transformed = torch.cat([
            #     torch.cat(
            #         [torch.sin((2**i) * pi * p),
            #          torch.cos((2**i) * pi * p)],
            #         dim=-1) for i in range(L)
            # ],
            #                           dim=-1)

            # ---> new code
            transform_to_cat = [
                torch.cat(
                    [torch.sin((2**i) * pi * p),
                     torch.cos((2**i) * pi * p)],
                    dim=-1) for i in range(L)
            ]
            p_transformed = torch.cat(transform_to_cat, dim=-1)

        return p_transformed

    def forward(self, p_in, ray_d, z_shape=None, z_app=None, **kwargs):
        a = F.relu
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(p_in.device)
        p = self.transform_points(p_in)
        net = self.fc_in(p)
        if z_shape is not None:
            net = net + self.fc_z(z_shape).unsqueeze(1)
        net = a(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app).unsqueeze(1)
        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_matrix()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r


class BoundingBoxGenerator(nn.Module):
    """Bounding box generator class.

    Args:
        n_boxes (int): number of bounding boxes (excluding background)
        scale_range_min (list): min scale values for x, y, z
        scale_range_max (list): max scale values for x, y, z
        translation_range_min (list): min values for x, y, z translation
        translation_range_max (list): max values for x, y, z translation
        z_level_plane (float): value of z-plane; only relevant if
            object_on_plane is set True
        rotation_range (list): min and max rotation value (between 0 and 1)
        check_collision (bool): whether to check for collisions
        collision_padding (float): padding for collision checking
        fix_scale_ratio (bool): whether the x/y/z scale ratio should be fixed
        object_on_plane (bool): whether the objects should be placed on a plane
            with value z_level_plane
        prior_npz_file (str): path to prior npz file (used for clevr) to sample
            locations from
    """
    def __init__(self,
                 n_boxes=1,
                 scale_range_min=[0.5, 0.5, 0.5],
                 scale_range_max=[0.5, 0.5, 0.5],
                 translation_range_min=[-0.75, -0.75, 0.],
                 translation_range_max=[0.75, 0.75, 0.],
                 z_level_plane=0.,
                 rotation_range=[0., 1.],
                 check_collison=False,
                 collision_padding=0.1,
                 fix_scale_ratio=True,
                 object_on_plane=False,
                 prior_npz_file=None,
                 **kwargs):
        super().__init__()

        self.n_boxes = n_boxes
        self.scale_min = torch.tensor(scale_range_min).reshape(1, 1, 3)
        self.scale_range = (torch.tensor(scale_range_max) -
                            torch.tensor(scale_range_min)).reshape(1, 1, 3)

        self.translation_min = torch.tensor(translation_range_min).reshape(
            1, 1, 3)
        self.translation_range = (torch.tensor(translation_range_max) -
                                  torch.tensor(translation_range_min)).reshape(
                                      1, 1, 3)

        self.z_level_plane = z_level_plane
        self.rotation_range = rotation_range
        self.check_collison = check_collison
        self.collision_padding = collision_padding
        self.fix_scale_ratio = fix_scale_ratio
        self.object_on_plane = object_on_plane

        if prior_npz_file is not None:
            try:
                prior = np.load(prior_npz_file)['coordinates']
                # We multiply by ~0.23 as this is multiplier of the original
                # clevr world and our world scale
                self.prior = torch.from_numpy(prior).float() * \
                    0.2378777237835723
            except Exception:
                print(
                    'WARNING: Clevr prior location file could not be loaded!')
                print('For rendering, this is fine, but for training, please '
                      'download the files using the download script.')
                self.prior = None
        else:
            self.prior = None

    def check_for_collison(self, s, t):
        n_boxes = s.shape[1]
        if n_boxes == 1:
            is_free = torch.ones_like(s[..., 0]).bool().squeeze(1)
        elif n_boxes == 2:
            d_t = (t[:, :1] - t[:, 1:2]).abs()
            d_s = (s[:, :1] + s[:, 1:2]).abs() + self.collision_padding
            is_free = (d_t >= d_s).any(-1).squeeze(1)
        elif n_boxes == 3:
            is_free_1 = self.check_for_collison(s[:, [0, 1]], t[:, [0, 1]])
            is_free_2 = self.check_for_collison(s[:, [0, 2]], t[:, [0, 2]])
            is_free_3 = self.check_for_collison(s[:, [1, 2]], t[:, [1, 2]])
            is_free = is_free_1 & is_free_2 & is_free_3
        else:
            print('ERROR: Not implemented')
        return is_free

    def get_translation(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        n_boxes = len(val)
        t = self.translation_min + \
            torch.tensor(val).reshape(1, n_boxes, 3) * self.translation_range
        t = t.repeat(batch_size, 1, 1)
        if self.object_on_plane:
            t[..., -1] = self.z_level_plane
        return t

    def get_rotation(self, batch_size=32, val=[0.]):
        r_range = self.rotation_range
        values = [r_range[0] + v * (r_range[1] - r_range[0]) for v in val]

        # ---> old code
        # r = torch.cat([
        #     get_rotation_matrix(value=v, batch_size=batch_size).unsqueeze(1)
        #     for v in values
        # ],
        #               dim=1)

        # ---> new code
        r_to_cat = [
            get_rotation_matrix(value=v, batch_size=batch_size).unsqueeze(1)
            for v in values
        ]
        r = torch.cat(r_to_cat, dim=1)

        r = r.float()
        return r

    def get_scale(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        n_boxes = len(val)
        if self.fix_scale_ratio:
            t = self.scale_min + \
                torch.tensor(val).reshape(
                    1, n_boxes, -1)[..., :1] * self.scale_range
        else:
            t = self.scale_min + \
                torch.tensor(val).reshape(1, n_boxes, 3) * self.scale_range
        t = t.repeat(batch_size, 1, 1)
        return t

    def get_random_offset(self, batch_size):
        n_boxes = self.n_boxes
        # Sample sizes
        if self.fix_scale_ratio:
            s_rand = torch.rand(batch_size, n_boxes, 1)
        else:
            s_rand = torch.rand(batch_size, n_boxes, 3)
        s = self.scale_min + s_rand * self.scale_range

        # Sample translations
        if self.prior is not None:
            idx = np.random.randint(self.prior.shape[0], size=(batch_size))
            t = self.prior[idx]
        else:
            t = self.translation_min + \
                torch.rand(batch_size, n_boxes, 3) * self.translation_range
            if self.check_collison:
                is_free = self.check_for_collison(s, t)
                while not torch.all(is_free):
                    t_new = self.translation_min + \
                        torch.rand(batch_size, n_boxes, 3) * \
                        self.translation_range
                    t[is_free == 0] = t_new[is_free == 0]
                    is_free = self.check_for_collison(s, t)
            if self.object_on_plane:
                t[..., -1] = self.z_level_plane

        def r_val():
            return self.rotation_range[0] + np.random.rand() * (
                self.rotation_range[1] - self.rotation_range[0])

        R = [
            torch.from_numpy(
                Rot.from_euler('z',
                               r_val() * 2 * np.pi).as_matrix())
            for i in range(batch_size * self.n_boxes)
        ]
        R = torch.stack(R, dim=0).reshape(batch_size, self.n_boxes,
                                          -1).cuda().float()
        return s, t, R

    def forward(self, batch_size=32):
        s, t, R = self.get_random_offset(batch_size)
        R = R.reshape(batch_size, self.n_boxes, 3, 3)
        return s, t, R
