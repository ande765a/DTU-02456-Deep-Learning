import torch
from skimage import transform

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, mask, has_subtitle = sample["image"], sample["mask"], sample["has_subtitle"]
        new_h, new_w = self.output_size
        
        image = transform.resize(image, (new_h, new_w))
        mask = transform.resize(mask, (new_h, new_w))
        
        return {
            "image": image,
            "mask": mask,
            "has_subtitle": has_subtitle
        }

class ToTensor(object):
    def __call__(self, sample):
        image, mask, has_subtitle = sample["image"], sample["mask"], sample["has_subtitle"]
    
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "has_subtitle": torch.tensor([has_subtitle])
        }