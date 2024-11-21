from PIL import Image
import numpy as np
import torch
import torchvision
import os
from typing import Union, Iterable


def to_numpy(obj: Union[torch.Tensor, np.ndarray]):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 3:
            obj = obj.unsqueeze(0)

        assert obj.ndim == 4, 'tensor should be 4 dimensional'
        obj = obj.permute(0, 2, 3, 1).cpu().numpy()
    elif isinstance(obj, np.ndarray):
        pass
    else:
        raise NotImplementedError('unknown instance to convert')

    if not np.issubdtype(obj.dtype, np.integer):
        obj = obj * 255

    return obj.astype(np.uint8)


def load_image(path: Union[str, list[str]], to_batch=False, device='cpu') -> torch.Tensor:
    to_tensor = torchvision.transforms.ToTensor()

    if isinstance(path, str):
        path = os.path.join(os.getcwd(), path)
        image = Image.open(path)
        image.convert('RGB')
        image = to_tensor(image)
    else:
        image = [to_tensor(Image.open(os.path.join(os.getcwd(), p)).convert('RGB')).unsqueeze(0) for p in path]
        image = torch.concat(image)

    if isinstance(path, str) and to_batch:
        image.unsqueeze_(0)

    return image.to(device)


# todo too duplicate, need to rewrite
def save_image(obj: Union[torch.Tensor, Image.Image, np.ndarray, list], fp: str, save_format='RGB', default_name='result.jpg'):
    fp = os.path.join(os.getcwd(), fp)

    if isinstance(obj, Image.Image):
        os.makedirs(os.path.dirname(fp))
        obj.convert(save_format).save(fp)
        return
    elif isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        obj = to_numpy(obj)
    elif isinstance(obj, list):
        pass
    else:
        raise NotImplementedError(f'not support type {obj.__class__.__name__}')

    if len(obj) == 1:
        if isinstance(obj[0], Image.Image):
            obj[0].convert(save_format).save(fp)
            return
        else:
            Image.fromarray(obj[0], mode=save_format).save(fp)
            return

    if os.path.isfile(fp):
        raise FileExistsError(f'{fp} is a file, but have {len(obj)} images to save')

    os.makedirs(fp, exist_ok=True)
    if isinstance(obj[0], torch.Tensor) or isinstance(obj[0], np.ndarray):
        images = [Image.fromarray(to_numpy(obj_), mode=save_format) for obj_ in obj]
    elif isinstance(obj[0], Image.Image):
        images = [image.convert(save_format) for image in obj]

    name, ext = os.path.splitext(default_name)
    for i, image in enumerate(images):
        image.save(os.path.join(fp, f'{name}_{i}{ext}'))

    return
