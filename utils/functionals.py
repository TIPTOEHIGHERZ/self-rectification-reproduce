import torch


def get_mask(image: torch.Tensor, thresh):
    mask = torch.zeros_like(image)
    # doing for all channels
    mask[image > thresh] = 1

    return mask

