import random
import torchvision.transforms.v2.functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from PIL import Image

class RandomHorizontalFlip(object):
    def __call__(self, images, label):
        if random.random() > 0.5:
            label = F.horizontal_flip(label)
            transformed_images = {}
            for key, value in images.items():
                flipped_images = [F.horizontal_flip(image) for image in value]
                transformed_images[key] = flipped_images
        return transformed_images, label

class RandomVerticalFlip(object):
    def __call__(self, images, label):
        if random.random() > 0.5:
            label = F.vertical_flip(label)
            transformed_images = {}
            for key, value in images.items():
                flipped_images = [F.vertical_flip(image) for image in value]
                transformed_images[key] = flipped_images
        return flipped_images, label

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, images, label):
        angle = random.randrange(-self.degrees, self.degrees, 90)
        label = F.rotate(label, angle)
        transformed_images = {}
        for key, value in images.items():
            flipped_images = [F.rotate(image) for image in value]
            transformed_images[key] = flipped_images
        return transformed_images, label


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

