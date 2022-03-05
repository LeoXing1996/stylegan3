import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

logger_py = logging.getLogger(__name__)


def arange_pixels(resolution=(128, 128),
                  batch_size=1,
                  image_range=(-1., 1.),
                  subsample_to=None,
                  invert_y_axis=False):
    """Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    the arranged pixels are in image (film) coordinates
                y (H)
                ^
                |
                |
    x (W) <------
    if invert_y_axis is true, y-axis should be pointed down.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    """
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack([pixel_locations[0], pixel_locations[1]],
                                  dim=-1).long().view(1, -1, 2).repeat(
                                      batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0
            and subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1],
                               size=(subsample_to, ),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    if invert_y_axis:
        assert (image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    """Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    """
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def transform_to_world(pixels,
                       depth,
                       camera_mat,
                       world_mat,
                       scale_mat=None,
                       invert=True,
                       use_absolute_depth=True):
    """Transforms pixel positions p with given depth value d to world
    coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    """
    assert (pixels.shape[-1] == 2)

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(camera_mat.shape[0], 1,
                                                     1).to(camera_mat.device)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    # [bz, H*W, 2] -> [bz, 2, H*W] (H*W=n_points)
    pixels = pixels.permute(0, 2, 1)
    # [bz, 2, H*W] -> [bz, 4, H*W]
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    # original pixels is in `screen` space,
    #   1. use (x, y) * depth to map (x, y) to camera space
    #   2. use 1 * depth to map (z) to camera space
    # depth: [bz, H*W, 1]
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels from camera to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    """Transforms world points to camera space.

    Args:
    p_world (tensor): world points tensor of size B x N x 3
    camera_mat (tensor): camera matrix
    world_mat (tensor): world matrix
    scale_mat (tensor): scale matrix
    """
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    p_world = torch.cat(
        [p_world, torch.ones(batch_size, n_p, 1).to(device)],
        dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    p_cam = camera_mat @ world_mat @ scale_mat @ p_world

    # Transform points back to 3D coordinates
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def origin_to_world(n_points,
                    camera_mat,
                    world_mat,
                    scale_mat=None,
                    invert=False):
    """Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: false)
    """
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1,
                                                     1).to(device)

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


def image_points_to_world(image_points,
                          camera_mat,
                          world_mat,
                          scale_mat=None,
                          invert=False,
                          negative_depth=True):
    """Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: False)
    """
    batch_size, n_pts, dim = image_points.shape
    assert (dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    if negative_depth:
        d_image *= -1.
    return transform_to_world(image_points,
                              d_image,
                              camera_mat,
                              world_mat,
                              scale_mat,
                              invert=invert)


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1 - t) * omega) / torch.sin(omega)
    s2 = torch.sin(t * omega) / torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z


def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi / 180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([[focal, 0., 0., 0.], [0., focal, 0., 0.],
                        [0., 0., 1, 0.], [0., 0., 0., 1.]]).reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat


def get_random_pose(range_u,
                    range_v,
                    range_radius,
                    batch_size=32,
                    invert=False):
    loc = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_radius[0] + \
        torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_middle_pose(range_u,
                    range_v,
                    range_radius,
                    batch_size=32,
                    invert=False):
    u_m, u_v, r_v = sum(range_u) * 0.5, sum(range_v) * \
        0.5, sum(range_radius) * 0.5
    loc = sample_on_sphere((u_m, u_m), (u_v, u_v), size=(batch_size))
    radius = torch.ones(batch_size) * r_v
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_camera_pose(range_u,
                    range_v,
                    range_r,
                    val_u=0.5,
                    val_v=0.5,
                    val_r=0.5,
                    batch_size=32,
                    invert=False):
    u0, ur = range_u[0], range_u[1] - range_u[0]
    v0, vr = range_v[0], range_v[1] - range_v[0]
    r0, rr = range_r[0], range_r[1] - range_r[0]
    u = u0 + val_u * ur
    v = v0 + val_v * vr
    r = r0 + val_r * rr

    loc = sample_on_sphere((u, u), (v, v), size=(batch_size))
    radius = torch.ones(batch_size) * r
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def to_sphere(u, v):
    r"""Convert u, v to sphere.
    phi: angle between y-axis (up axis)
    theta: angle between x-axis
    """
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def sample_on_sphere(
        range_u=(0, 1), range_v=(0, 1), size=(1, ), to_pytorch=True):
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)

    sample = to_sphere(u, v)
    if to_pytorch:
        sample = torch.tensor(sample).float()

    return sample


def look_at(eye,
            at=np.array([0, 0, 0]),
            up=np.array([0, 0, 1]),
            eps=1e-5,
            to_pytorch=True):
    """Get view projection matrix (world to camera).

        eye: position of camera
        at: position of object
        up: up direction of the camera. denotes the camera is up or down.
            This direction will be revised by this function.

    Returns
        [[x, y, z]]: Actually the inverse of lookAt (camera to world).
        The look at should be cat at axis=1
    """
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    # z / forward direction
    z_axis = eye - at
    z_axis /= np.max(
        np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    # y \times z = x / right direction
    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(
        np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    # z \times x = y / up direction
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(
        np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    # ---> old code
    # r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(
    #     -1, 3, 1), z_axis.reshape(-1, 3, 1)),
    #                        axis=2)

    # ---> new code
    x_axis_ = x_axis.reshape(-1, 3, 1)
    y_axis_ = y_axis.reshape(-1, 3, 1)
    z_axis_ = z_axis.reshape(-1, 3, 1)
    r_mat = np.concatenate((x_axis_, y_axis_, z_axis_), axis=2)

    if to_pytorch:
        r_mat = torch.tensor(r_mat).float()

    return r_mat


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_matrix()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r
