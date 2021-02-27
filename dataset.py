#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
from transforms import Resize, Rotate, HorizontalFlip, VerticalFlip,\
     Normalize, ToTensor
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import random
from random import shuffle
import os
import numpy as np
import torch


#-----------------------------------------------------------------------#
#                             partitioning                              #
#       Splits the image paths into 3 groups: train, val, and test      #
#-----------------------------------------------------------------------#
# Returns a dictionary with 3 keys ['train', 'val', 'test'], where the  #
# values are lists of image paths (jpg).                                #
# The paths are stratified shuffled split.                              #
#-----------------------------------------------------------------------#
# split_ratio:      A list of the (train,val,test) split ratio,        #
#                   e.g. [0.7, 0.1, 0.2].                               #
# partition:        A dictionary of train, valid, test image paths      #
# classes:          A list of image labels                              #    
# images_path:      A dictionary of image paths for each image label    #
# l:                The total number of paths in a class                #
# split_pt:         split the length of path list using the split_ratio.#
# train, val, test: Dictionaries of train, valid, test image paths for  #
#                   each image label to be used for stratify split.     #
#-----------------------------------------------------------------------#
def partitioning(split_ratio):
    images_path = {}
    partition = {'train':[], 'val':[], 'test':[]}
    classes =['Blowhole', 'Crack','Free']

    train = {}
    val = {}
    test = {}
    
    for c in classes:
        # retreive image file paths recursively
        images_path [c] = glob('data/MT_' + c + '/Imgs/*.jpg',recursive=True)    
        
        # shuffle the path list 
        random.shuffle(images_path[c])
        # undersample the Free set
        if c== 'Free':
            images_path [c] = images_path [c] [:80]
            
        l = len(images_path[c])
        
        # find the split indices
        split_pt = [int(split_ratio[0]*l), int((split_ratio[0]+ split_ratio[1])*l)]
        
        # stratify split the paths
        train [c] = images_path[c][:split_pt[0]]
        val [c] = images_path[c][split_pt[0]: split_pt[1]]
        test [c] = images_path[c][split_pt[1]:]
        print(c, '_ ', 'train: ', len(train[c]), ' ', 'val: ', len(val[c]), ' ', \
              'test: ', len(test[c]), ' ', 'total: ', len(train[c]) + len(val[c])\
              + len(test[c]))
        
        # append the class paths to their partitions
        partition['train'].extend(train[c])
        partition['val'].extend(val[c])
        partition['test'].extend(test[c])
        
    # return the partitions    
    return partition


#-----------------------------------------------------------------------#
#                   SurfaceDefectDetectionDataset                       #
#                        constructs datasets                            #
#-----------------------------------------------------------------------#
# returns: a dataset of 2D image and mask torch tensor pairs            #
#-----------------------------------------------------------------------#
# images_path_list: The list of image files path. They have jpg ext.    #
# partition:        Either train, valid, or test                        #
# no_transform:     False if augmentation is required.                  #
#-----------------------------------------------------------------------#
class SurfaceDefectDetectionDataset(Dataset):    
    def __init__(self, images_path_list, partition, no_transform =  False):
        super().__init__()
        self.images_path_list = images_path_list
        self.partition = partition
        self.no_transform = no_transform

    @staticmethod
    def transform(image, mask, p, no_transform=False):
        """
        Args:
            image:        image in PIL
            mask:         mask in PIL
            p:            Type of partition, either train, valid, or test
            no_transform: False if augmentation is required   
        Returns:
            image and mask pair in torch tensor.
        """
        # Resize
        t1 = Resize((224,224))
        image = t1(image)
        mask = t1(mask)

        if p == 'train' and no_transform == False:
            # Random horizontal flipping with 50% probability
            if random.random() > 0.5:
                t2 = HorizontalFlip()
                image = t2(image)
                mask = t2(mask)

            # Random vertical flipping with 50% probability
            if random.random() > 0.5:
                t3 = VerticalFlip()
                image = t3(image)
                mask = t3(mask)

            # Rotate
            angle = random.choice([0, -90, 90, 180])
            t4 = Rotate(angle)
            image = t4(image)
            mask = t4(mask)
          
        # Transform to tensor
        t5 = ToTensor()
        image = t5(image)
        mask = t5(mask)

        # Threshold mask, threshold limit is 0.5
        mask = (mask.cpu().numpy() >= 0.5)*1
        mask = torch.from_numpy(mask).float().cuda()

        return image, mask

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        # Generate one batch of data
        # Open the image file which is in jpg     
        image = Image.open(self.images_path_list[idx])
        # The mask is in png. 
        # Use the image path, and change its extension to png to get the mask's path.
        mask = Image.open(os.path.splitext(self.images_path_list[idx])[0]+'.png') 
        
        # Transform the image and mask PILs to torch tensors. 
        # Perform augmentation if required.
        image, mask = self.transform(image, mask, self.partition, self.no_transform)
        
        #return the image and mask pair tensors
        return image, mask