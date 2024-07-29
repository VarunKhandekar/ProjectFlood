import random
import torchvision.transforms.v2.functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from PIL import Image

class RandomHorizontalFlip(object):
    def __call__(self, images, label):
        if random.random() > 0.5:
            image = F.horizontal_flip(image)
            label = F.horizontal_flip(label)
        return image, label

class RandomVerticalFlip(object):
    def __call__(self, images, label):
        if random.random() > 0.5:
            image = F.vertical_flip(image)
            label = F.vertical_flip(label)
        return image, label

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, images, label):
        angle = random.randrange(-self.degrees, self.degrees+1, 90)
        image = F.rotate(image, angle)
        label = F.rotate(label, angle)
        return images, label


# class RandomResizedCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, image, label):
#         i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(3./4., 4./3.))
#         image = F.resized_crop(image, i, j, h, w, self.size)
#         label = F.resized_crop(label, i, j, h, w, self.size)
#         return image, label

# class ToTensor(object):
#     def __call__(self, image, label):
#         image = F.to_tensor(image)
#         label = torch.tensor(label, dtype=torch.long)
#         return image, label
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

