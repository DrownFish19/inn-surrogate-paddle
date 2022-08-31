import math
import pathlib
import warnings
from typing import Union, Optional, List, Tuple, Text, BinaryIO

import matplotlib.pyplot as plt
import paddle
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

plt.switch_backend('agg')


@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]], nrow: int = 8, padding: int = 2, normalize: bool = False,
              value_range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0, **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or (isinstance(tensor, list) and all(isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(tensor: Union[paddle.Tensor, List[paddle.Tensor]], fp: Union[Text, pathlib.Path, BinaryIO], format: Optional[str] = None,
               **kwargs) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose([1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def save_samples(save_dir, images, epoch, layer, plot, name, nrow=4, heatmap=True, cmap='jet'):
    """Save samples in grid as images or plots
    Args:
        images (Tensor): B x C x H x W
    """

    if images.shape[0] < 10:
        nrow = 2
        ncol = images.shape[0] // nrow
    else:
        ncol = nrow

    if heatmap:
        for c in range(images.shape[1]):
            # (11, 12)
            fig = plt.figure(1, (12, 12))
            grid = ImageGrid(fig, 111, nrows_ncols=(nrow, ncol), axes_pad=0.3, share_all=False, cbar_location="right", cbar_mode="single",
                             cbar_size="3%", cbar_pad=0.1)
            for j, ax in enumerate(grid):
                im = ax.imshow(images[j][c], cmap='jet', origin='lower', interpolation='bilinear')
                if j == 0:
                    ax.set_title('actual')
                elif j == 1:
                    ax.set_title('mean')
                else:
                    ax.set_title('sample %d' % (j - 1))
                ax.set_axis_off()
                ax.set_aspect('equal')
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.toggle_label(True)
            plt.subplots_adjust(top=0.95)
            plt.savefig(save_dir + '/{}_c{}_epoch{}_layer{}.pdf'.format(name, c, epoch, layer), bbox_inches='tight')
            plt.close(fig)
    else:
        save_image(images, save_dir + '/fake_samples_epoch_{}.png'.format(epoch), nrow=nrow, normalize=True)
