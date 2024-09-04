import random
import torchvision.transforms.v2.functional as F

class RandomHorizontalFlip(object):
    '''
    Randomly horizontally flip label and all input images.
    '''
    def __call__(self, images, label):
        transformed_images = {}
        if random.random() > 0.5:
            transformed_label = F.horizontal_flip(label)
            for key, value in images.items():
                edited_images = [F.horizontal_flip(image) for image in value]
                transformed_images[key] = edited_images
            return transformed_images, transformed_label
        return images, label      


class RandomVerticalFlip(object):
    '''
    Randomly vertically flip label and all input images.
    '''
    def __call__(self, images, label):
        transformed_images = {}
        if random.random() > 0.5:
            transformed_label = F.vertical_flip(label)
            for key, value in images.items():
                edited_images = [F.horizontal_flip(image) for image in value]
                transformed_images[key] = edited_images
            return transformed_images, transformed_label
        return images, label
        


class RandomRotation(object):
    '''
    Randomly rotate label and all input images (90 degree multiples).
    '''
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, images, label):
        angle = random.randrange(-self.degrees, self.degrees, 90)
        transformed_label = F.rotate(label, angle)
        transformed_images = {}
        for key, value in images.items():
            edited_images = [F.rotate(image, angle) for image in value]
            transformed_images[key] = edited_images
        return transformed_images, transformed_label


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
    '''
    Create a set of sequential random transformations to be applied to label and all input images.
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, label):
        for t in self.transforms:
            images, label = t(images, label)
        return images, label
    
train_transform = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(180),
])