# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Tool for creating ZIP/PNG based datasets."""

import functools
import io
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import mmcv
import numpy as np
import PIL.Image
from mmcv.fileio import FileClient
from tqdm import tqdm

# ---------------------------------------------------------------------------


def error(msg):
    print('Error: ' + msg)
    sys.exit(1)


# ---------------------------------------------------------------------------


def parse_tuple(s: str) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    """
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ---------------------------------------------------------------------------


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a


# ---------------------------------------------------------------------------


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]


# ---------------------------------------------------------------------------


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION  # type: ignore


# ---------------------------------------------------------------------------


def make_transform(
    transform: Optional[str], output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:

    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2:(img.shape[0] + crop) // 2,
                  (img.shape[1] - crop) // 2:(img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2:(img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2:(width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error(f'must specify --resolution=WxH when using {transform} '
                  'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error(f'must specify --resolution=WxH when using {transform} '
                  ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'


# ---------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--source',
              help='Directory or archive name for input dataset',
              required=True,
              metavar='PATH')
@click.option('--dest',
              help='Output directory or archive name for output dataset',
              required=True,
              metavar='PATH')
@click.option('--max-images',
              help='Output only up to `max-images` images',
              type=int,
              default=None)
@click.option('--transform',
              help='Input crop/resize mode',
              type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution',
              help='Output resolution (e.g., \'512x512\')',
              metavar='WxH',
              type=parse_tuple)
def convert_dataset(ctx: click.Context, source: str, dest: str,
                    max_images: Optional[int], transform: Optional[str],
                    resolution: Optional[Tuple[int, int]]):
    """Convert an image dataset into a dataset archive usable with StyleGAN2
    ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip  # noqa

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompressed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

    client = FileClient(backend='petrel')
    assert client.isdir(source), f'source dir \'{source}\' is invalid'
    PIL.Image.init()  # type: ignore

    suffix = tuple(PIL.Image.EXTENSION.keys())
    fname_all = [
        fname for fname in client.list_dir_or_file(
            source, list_dir=False, recursive=True, suffix=suffix)
    ]

    num_files = len(fname_all)

    if resolution is None:
        resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    labels = []
    for idx, fname in tqdm(enumerate(fname_all), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        fname_full = os.path.join(source, fname)
        img_bytes = client.get(fname_full)
        img_orig = mmcv.imfrombytes(img_bytes,
                                    flag='color',
                                    channel_order='rgb',
                                    backend='pillow')
        img = transform_image(img_orig)

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error('Image dimensions after scale and crop are required '
                      f'to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2**int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required '
                      'to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [
                f'  dataset {k}/cur image {k}: '
                f'{dataset_attrs[k]}/{cur_image_attrs[k]}'
                for k in dataset_attrs.keys()
            ]
            error(f'Image {archive_fname} attributes must be equal across '
                  'all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        client.put(image_bits.getbuffer().tobytes(),
                   os.path.join(dest, archive_fname))
        labels.append([archive_fname, fname['label']]
                      if fname['label'] is not None else None)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    convert_dataset()  # pylint: disable=no-value-for-parameter
