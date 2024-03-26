from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output: torch.Tensor, gt: torch.Tensor, agg="mean") -> torch.Tensor:
    """
    Computes the L1 loss, which is the mean absolute error between the network output and the ground truth.

    Args:
        network_output: The output from the network.
        gt: The ground truth tensor.
        agg: The aggregation method to be used. Defaults to "mean".
    Returns:
        The computed L1 loss.
    """
    l1_loss = torch.abs(network_output - gt)
    if agg == "mean":
        return l1_loss.mean()
    elif agg == "sum":
        return l1_loss.sum()
    elif agg == "none":
        return l1_loss
    else:
        raise ValueError("Invalid aggregation method.")


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size: The size of the window for the Gaussian kernel.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The 1D Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    """
    Creates a 2D Gaussian window/kernel for SSIM computation.

    Args:
        window_size: The size of the window to be created.
        channel: The number of channels in the image.

    Returns:
        A 2D Gaussian window expanded to match the number of channels.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window_size: The size of the window to be used in SSIM computation. Defaults to 11.
        size_average: If True, averages the SSIM over all pixels. Defaults to True.

    Returns:
        The computed SSIM value.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: Variable, window_size: int,
          channel: int, size_average: bool = True) -> torch.Tensor:
    """
    Internal function to compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window: The Gaussian window/kernel for SSIM computation.
        window_size: The size of the window to be used in SSIM computation.
        channel: The number of channels in the image.
        size_average: If True, averages the SSIM over all pixels.

    Returns:
        The computed SSIM value.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def isotropic_loss(scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes loss enforcing isotropic scaling for the 3D Gaussians
    Args:
        scaling: scaling tensor of 3D Gaussians of shape (n, 3)
    Returns:
        The computed isotropic loss
    """
    mean_scaling = scaling.mean(dim=1, keepdim=True)
    isotropic_diff = torch.abs(scaling - mean_scaling * torch.ones_like(scaling))
    return isotropic_diff.mean()
