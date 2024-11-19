from PIL import Image
import numpy as np
import torch
import torchvision
import os
from typing import Union, Iterable


def to_numpy(obj: Union[torch.Tensor, np.ndarray]):
    if isinstance(obj, torch.Tensor):
        assert obj.ndim == 4, 'tensor should be 4 dimensional'
        obj = obj.permute(0, 2, 3, 1).cpu().numpy()
        return obj
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise NotImplementedError('unknown instance to convert')


def load_image(path: str) -> torch.Tensor:
    path = os.path.join(os.getcwd(), path)
    image = Image.open(path)
    image.convert('RGB')
    image = torchvision.transforms.ToTensor()(image)

    return image


def save_image(obj: Union[torch.Tensor, Image.Image, np.ndarray, list], fp: str):
    fp = os.path.join(os.getcwd(), fp)

    if isinstance(obj, torch.Tensor):
        obj = to_numpy(obj)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = to_numpy(obj[i])
        obj = np.concatenate(obj, axis=0)
    elif isinstance(obj, Image.Image):
        if os.path.exists(fp) and os.path.isdir(fp):
            raise ValueError('fp is not a file path')
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        obj.save(fp)
        return

    os.makedirs(fp, exist_ok=True)
    for i in range(len(obj)):
        image = Image.fromarray(obj[i], mode='RGB')
        image.save(os.path.join(fp, f'image_{i}.jpg'))
    return
