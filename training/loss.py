# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------


class Loss:

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain,
                             cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


#----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):

    def __init__(self,
                 device,
                 G,
                 D,
                 augment_pipe=None,
                 r1_gamma=10,
                 style_mixing_prob=0,
                 pl_weight=0,
                 pl_batch_shrink=2,
                 pl_decay=0.01,
                 pl_no_weight_grad=False,
                 blur_init_sigma=0,
                 blur_fade_kimg=0,
                 D_obj=None):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

        # attribute for obj generator
        self.D_obj = D_obj
        self.pl_mean_obj = torch.zeros([], device=device)

    def run_G(self, z, c, update_emas=False, nerf_kwargs=None):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64,
                                     device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z),
                                                c,
                                                update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws,
                               update_emas=update_emas,
                               nerf_kwargs=nerf_kwargs)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False, is_obj=False):
        if is_obj:
            assert self.D_obj is not None
            D = self.D_obj
        else:
            D = self.D

        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(
                    -blur_size, blur_size + 1,
                    device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain,
                             cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(
            1 - cur_nimg / (self.blur_fade_kimg * 1e3),
            0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # G_obj
        if phase in ['Gobj_main', 'Gobj_both']:
            with torch.autograd.profiler.record_function('Gobj_forward'):
                nerf_kwargs = dict(not_render_background=True)
                gen_img, _ = self.run_G(gen_z, gen_c, nerf_kwargs=nerf_kwargs)
                gen_logits = self.run_D(gen_img, gen_c, is_obj=True)
                training_stats.report('Loss/scores/fake_obj', gen_logits)
                training_stats.report('Loss/signs/fake_obj', gen_logits.sign())
                loss_Gobj = torch.nn.functional.softplus(
                    -gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G_obj/loss', loss_Gobj)
            with torch.autograd.profiler.record_function('Gobj_backward'):
                loss_Gobj.mean().mul(gain).backward()

        if phase in ['Gobj_main', 'Gobj_reg']:
            with torch.autograd.profiler.record_function('Gobj_pl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                nerf_kwargs = dict(not_render_background=True)
                gen_img, gen_ws = self.run_G(gen_z[:batch_size],
                                             gen_c[:batch_size], nerf_kwargs=nerf_kwargs)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(
                    gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function(
                        'pl_obj_grads'), conv2d_gradfix.no_weight_gradients(
                            self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[
                        (gen_img * pl_noise).sum()
                    ],
                                                   inputs=[gen_ws],
                                                   create_graph=True,
                                                   only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean_obj.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean_obj.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_obj_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G_obj/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gobj_pl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(
                    -gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size],
                                             gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(
                    gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function(
                        'pl_grads'), conv2d_gradfix.no_weight_gradients(
                            self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[
                        (gen_img * pl_noise).sum()
                    ],
                                                   inputs=[gen_ws],
                                                   create_graph=True,
                                                   only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # D_obj
        loss_Dgen = 0
        if phase in ['Dmain_obj', 'Dboth_obj']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                nerf_kwargs = dict('')
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True, )
                gen_logits = self.run_D(gen_img,
                                        gen_c,
                                        blur_sigma=blur_sigma,
                                        update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(
                    gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img,
                                        gen_c,
                                        blur_sigma=blur_sigma,
                                        update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(
                    gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(
                    phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp,
                                         real_c,
                                         blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(
                        -real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss',
                                          loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function(
                            'r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()


#----------------------------------------------------------------------------


class NeRFLoss(Loss):
    """NOTE: this class is added by us.
    This class perform loss calculation related to nerf model
    such as foreground loss
    """

    def __init__(self,
                 device,
                 G,
                 D,
                 augment_pipe=None,
                 r1_gamma=10,
                 style_mixing_prob=0,
                 pl_weight=0,
                 pl_batch_shrink=2,
                 pl_decay=0.01,
                 pl_no_weight_grad=False,
                 blur_init_sigma=0,
                 blur_fade_kimg=0):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        nerf_kwargs = dict(not_render_background=True)
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64,
                                     device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z),
                                                c,
                                                update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws,
                               update_emas=update_emas,
                               nerf_kwargs=nerf_kwargs)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False, is_obj=False):
        if is_obj:
            assert self.D_obj is not None
            D = self.D_obj
        else:
            D = self.D

        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(
                    -blur_size, blur_size + 1,
                    device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain,
                             cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(
            1 - cur_nimg / (self.blur_fade_kimg * 1e3),
            0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward_obj'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake_obj', gen_logits)
                training_stats.report('Loss/signs/fake_obj', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(
                    -gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss_obj', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward_obj'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward_obj'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size],
                                             gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(
                    gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function(
                        'pl_grads'), conv2d_gradfix.no_weight_gradients(
                            self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[
                        (gen_img * pl_noise).sum()
                    ],
                                                   inputs=[gen_ws],
                                                   create_graph=True,
                                                   only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty_obj', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg_obj', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward_obj'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward_obj'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img,
                                        gen_c,
                                        blur_sigma=blur_sigma,
                                        update_emas=True)
                training_stats.report('Loss/scores/fake_obj', gen_logits)
                training_stats.report('Loss/signs/fake_obj', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(
                    gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward_obj'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward_obj'):
                real_img_tmp = real_img.detach().requires_grad_(
                    phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp,
                                         real_c,
                                         blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real_obj', real_logits)
                training_stats.report('Loss/signs/real_obj', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(
                        -real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_obj',
                                          loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function(
                            'r1_grads_obj'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_obj', r1_penalty)
                    training_stats.report('Loss/D/reg_obj', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward_obj'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
