from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn as nn
import random


class Resize(TF):
    """Resize the image to a given size. """
    def __init__(self):
        super(Resize, self).__init__()
        self.output_size = (224,224)

    def __call__(self, img, output_size ):
        """
        Args:
            img (PIL Image): Image to be resized.
            output_size (tuple): Desired output size.
        """
        assert isinstance(output_size, tuple)
        return TF.resize(img, output_size)


class Rotate(TF):
    """Rotate the image by the given angle."""
    def __init__(self):
        super(Rotate, self).__init__()

    def __call__(self, img, angle):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
            angle: Desired rotation angle in degrees
        """
        return TF.rotate(img, angle)


class VerticalFlip(TF):
    """Vertically flip the image."""
    def __init__(self):
        super(VerticalFlip, self).__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        """
        return TF.vflip(img)

class HorizontalFlip(TF):
    """Horizontally flip the image."""
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        """
        return TF.hflip(img)

class Normalize(TF):
    """Convert the color range of a grayscale image to [0,1] and normalize it using mean and standard deviation."""        
    def __init__(self):
        super(Normalize, self).__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be normalized.
        """
        # scale colour range from [0, 255] to [0, 1]
        img = img.astype(np.float32)/255

        # compute the mean and standard deviation of the image
        return TF.normalize(img, np.mean(img), np.std(img))
        

class ToTensor(TF):
    """Convert ndarrays in to Tensors."""
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be converted to tensor.
        """
        return TF.ToTensor(img)  


