import argparse
from math import sqrt

import os
import os.path as osp
import imageio
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid

import dnnlib
import legacy
from torch_utils import misc


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1 - t) * omega) / torch.sin(omega)
    s2 = torch.sin(t * omega) / torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z


class NeRFRenderer(object):

    def __init__(self, generator, seed=42, root='vis_out', device='cuda:0'):
        self.generator = generator
        self.synthesis_network = self.generator.synthesis
        self.nerf = self.synthesis_network.nerf

        self.seed = seed
        self.root = root
        self.device = device

        self.sample_tmp = 0.65  # TODO: what's this?

    @staticmethod
    def norm_output(out, drange=[-1, 1]):
        lo, hi = drange
        out = (out - lo) * (255 / (hi - lo))
        img = out.round().clip(0, 255)
        return img

    def get_stylegan_noise(self, batch_size):
        z = torch.from_numpy(
            np.random.RandomState(self.seed).randn(
                batch_size, self.generator.z_dim)).to(self.device)
        return z

    def render_add_cars(self, batch_size):

        nerf = self.nerf

        # generate stylegan noise
        z = self.get_stylegan_noise(batch_size)

        # Get NeRF values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = nerf.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = nerf.sample_z(z_shape_obj[:, :1].repeat(1, 6, 1).shape,
                                    tmp=self.sample_tmp)
        z_app_obj = nerf.sample_z(z_app_obj[:, :1].repeat(1, 6, 1).shape,
                                  tmp=self.sample_tmp)
        bg_rotation = nerf.get_random_bg_rotation(batch_size)
        camera_matrices = nerf.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
        ]

        t = [
            [-0.7, -.8, 0.],
            [-0.7, 0.5, 0.],
            [-0.7, 1.8, 0.],
            [1.5, -.8, 0.],
            [1.5, 0.5, 0.],
            [1.5, 1.8, 0.],
        ]
        r = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        outs = []
        for i in range(1, 7):
            transformations = nerf.get_transformations(s[:i], t[:i], r[:i],
                                                       batch_size)
            latent_codes = [
                z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg, z_app_bg
            ]
            with torch.no_grad():
                nerf_kwargs = dict(latent_codes=latent_codes,
                                   camera_matrices=camera_matrices,
                                   transformations=transformations,
                                   bg_rotation=bg_rotation,
                                   mode='val')
                out = self.generator(z=z, c=None, nerf_kwargs=nerf_kwargs)

                out = self.norm_output(out)

            outs.append(out)
        outs = torch.stack(outs)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = outs[[idx]]

        save_root = osp.join(self.root, 'render_add_cars')
        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, f'{self.seed}.mp4')
        self.save_video(save_path, outs)

    def render_object_rotation(self, batch_size=15, n_steps=32):
        nerf = self.nerf
        # gen = self.generator
        bbox_generator = nerf.bounding_box_generator

        n_boxes = bbox_generator.n_boxes

        # generate stylegan noise
        z = self.get_stylegan_noise(batch_size)

        # import ipdb
        # ipdb.set_trace()
        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Get Random codes and bg rotation
        latent_codes = nerf.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = nerf.get_random_bg_rotation(batch_size)

        # Set Camera
        camera_matrices = nerf.get_camera(batch_size=batch_size)
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]
        s, t, _ = nerf.get_transformations(s_val, t_val, r_val, batch_size)

        out = []
        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = nerf.get_rotation(r, batch_size)

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                nerf_kwargs = dict(latent_codes=latent_codes,
                                   camera_matrices=camera_matrices,
                                   transformations=transformations,
                                   bg_rotation=bg_rotation,
                                   mode='val')
                out_i = self.generator(z=z, c=None, nerf_kwargs=nerf_kwargs)
                out_i = self.norm_output(out_i)
            out.append(out_i.cpu())

        out = torch.stack(out)

        save_root = osp.join(self.root, 'rotation_object')
        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, f'{self.seed}.mp4')
        self.save_video(save_path, out)

    def render_interplolation_bg(self,
                                 batch_size=15,
                                 n_samples=6,
                                 n_steps=32,
                                 mode='app'):
        nerf = self.nerf
        n_boxes = nerf.bounding_box_generator.n_boxes

        z = self.get_stylegan_noise(batch_size)

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            nerf.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            nerf.sample_z(z_app_bg_1.shape, tmp=self.sample_tmp)
            for j in range(n_samples)
        ]

        bg_rotation = nerf.get_random_bg_rotation(batch_size)
        camera_matrices = nerf.get_camera(batch_size=batch_size)

        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = nerf.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)], t_val,
            [0.5 for i in range(n_boxes)], batch_size)

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j + 1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [
                        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_ii
                    ]
                else:
                    latent_codes = [
                        z_shape_obj_1, z_app_obj_1, z_ii, z_app_bg_1
                    ]
                with torch.no_grad():
                    nerf_kwargs = dict(latent_codes=latent_codes,
                                       camera_matrices=camera_matrices,
                                       transformations=transformations,
                                       bg_rotation=bg_rotation,
                                       mode='val')
                    out_i = self.generator(z=z,
                                           c=None,
                                           nerf_kwargs=nerf_kwargs)
                    out_i = self.norm_output(out_i)
                out.append(out_i.cpu())
        out = torch.stack(out)

        save_root = osp.join(self.root, f'interpolation_bg_{mode}')
        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, f'{self.seed}.mp4')
        self.save_video(save_path, out)

    def render_object_translation_horizontal(self, batch_size=15, n_steps=32):
        nerf = self.nerf

        z = self.get_stylegan_noise(batch_size)

        # Get values
        latent_codes = nerf.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = nerf.get_random_bg_rotation(batch_size)
        camera_matrices = nerf.get_camera(batch_size=batch_size)
        n_boxes = nerf.bounding_box_generator.n_boxes
        s = [[0., 0., 0.] for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[x_val, i, 0.]]
            transformations = nerf.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                nerf_kwargs = dict(latent_codes=latent_codes,
                                   camera_matrices=camera_matrices,
                                   transformations=transformations,
                                   bg_rotation=bg_rotation,
                                   mode='val')

                out_i = self.generator(z=z, c=None, nerf_kwargs=nerf_kwargs)
                out_i = self.norm_output(out_i)
            out.append(out_i.cpu())
        out = torch.stack(out)

        save_root = osp.join(self.root, 'translation_horizontal')
        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, f'{self.seed}.mp4')
        self.save_video(save_path, out)

    @staticmethod
    def save_video(out_file,
                   img_list,
                   n_row=5,
                   add_reverse=False,
                   save_imgs=False):
        n_steps, batch_size = img_list.shape[:2]
        nrow = n_row if (n_row is not None) else int(sqrt(batch_size))
        img = [
            (make_grid(img, nrow=nrow,
                       pad_value=1.).permute(1, 2,
                                             0)).cpu().numpy().astype(np.uint8)
            for img in img_list
        ]
        if add_reverse:
            img += list(reversed(img))

        # if save_imgs:
        #     for idx in range():
        #         pass
        imageio.mimwrite(out_file, img, fps=30, quality=8)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_config', type=str)
    parser.add_argument('--resume_pkl', type=str)
    return parser.parse_args()


def main():
    import sys
    sys.path.append('training')
    args = get_args()
    res = 256
    cbase = 32768
    cmax = 512
    G_kwargs = dnnlib.EasyDict(class_name=None,
                               z_dim=512,
                               w_dim=512,
                               mapping_kwargs=dnnlib.EasyDict())
    G_kwargs.channel_base = cbase
    G_kwargs.channel_max = cmax
    G_kwargs.mapping_kwargs.num_layers = 8

    G_kwargs.fused_modconv_default = 'inference_only'

    G_kwargs.class_name = 'training.networks_stylegan2.Generator_with_NeRF'

    nerf_config = args.nerf_config
    with open(nerf_config, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    nerf_kwargs = dnnlib.EasyDict()
    G_kwargs.nerf_kwargs = cfg_special.get('nerf_kwargs', nerf_kwargs)

    decoder_kwargs = dnnlib.EasyDict()
    G_kwargs.decoder_kwargs = cfg_special.get('decoder_kwargs', decoder_kwargs)

    bg_decoder_kwargs = dnnlib.EasyDict()
    G_kwargs.bg_decoder_kwargs = cfg_special.get('bg_decoder_kwargs',
                                                 bg_decoder_kwargs)

    bbox_kwargs = dnnlib.EasyDict()
    G_kwargs.bbox_kwargs = cfg_special.get('bbox_kwargs', bbox_kwargs)

    common_kwargs = dict(c_dim=0, img_resolution=res, img_channels=3)

    G = dnnlib.util.construct_class_by_name(
        **G_kwargs, **common_kwargs).eval().requires_grad_(False)

    with dnnlib.util.open_url(args.resume_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
        misc.copy_params_and_buffers(resume_data['G_ema'], G)

    renderer = NeRFRenderer(G.cuda())
    # renderer.render_add_cars(batch_size=5)
    # renderer.render_object_rotation(batch_size=20)
    # renderer.render_interplolation_bg(batch_size=20, mode='app')
    # renderer.render_interplolation_bg(batch_size=20, mode='shape')
    renderer.render_object_translation_horizontal(batch_size=5)


if __name__ == '__main__':
    main()
