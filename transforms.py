#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import random
import numpy as np


class Resize(object):
    """Resize the image to a given size. """
    def __init__(self, output_size):
        """
        Args:
            output_size (tuple): Desired output size.
        """
        super().__init__()
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be resized.
        """

        return TF.resize(img, self.output_size)


class Rotate(object):
    """Rotate the image by the given angle."""
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
            angle: Desired rotation angle in degrees
        """
        return TF.rotate(img, self.angle)


class VerticalFlip(object):
    """Vertically flip the image."""
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        """
        return TF.vflip(img)

class HorizontalFlip(object):
    """Horizontally flip the image."""
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        """
        return TF.hflip(img)

class Normalize(object):
    """
    Convert the color range of a grayscale image to [0,1], and
    normalize it using mean and standard deviation.
    """        
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        """
        Args:
            img (tensor image): Image to be normalized.
        """
        # scale colour range from [0, 255] to [0, 1]
        img = img/255.0

        return TF.normalize(img, 0, 1)      

class ToTensor(object):
    """Convert PIL Image in to Tensors."""
    def __init__(self):
        super().__init__()

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be converted to tensor.
        """
        return TF.to_tensor(img)  


