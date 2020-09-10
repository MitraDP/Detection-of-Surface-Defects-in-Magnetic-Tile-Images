from my_classes.transforms import Resize, Rotate, HorizontalFlip, VerticalFlip, Normalize, ToTensor
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import random
from random import shuffle
import os

def partitioning (split_ratio):
    images_path = {}
    partition = {'train':[], 'val':[], 'test':[]}
    classes =['Blowhole', 'Crack', 'Break', 'Fray', 'Uneven', 'Free']

    train = {}
    val = {}
    test = {}
    #split_ratio = [0.60, 0.20, 0.20]
    for c in classes:
        images_path[c] = glob('data/MT_' + c + '/Imgs/*.jpg',recursive=True)
        random.shuffle(images_path[c])
        l = len(images_path[c])
        split_pt = [int(split_ratio[0]*l), int((split_ratio[0]+ split_ratio[1])*l)]
        #print(c , ': ', l)
        train [c] = images_path[c][:split_pt[0]]
        val [c] = images_path[c][split_pt[0]: split_pt[1]]
        test [c] = images_path[c][split_pt[1]:]
        print(c,'_ ','train: ', len(train[c]),' ','val: ', len(val[c]),' ','test: ', len(test[c]),' ','total: ',len(train[c])+len(val[c])+len(test[c]))
        partition['train'].extend(train[c])
        partition['val'].extend(val[c])
        partition['test'].extend(test[c])
    return partition

class SurfaceDefectDetectionDataset(Dataset):
    
    def __init__(self, images_path_list):
        """
        Args:
            images_path_list: list of image files path.
        """
        super().__init__()
        self.images_path_list = images_path_list

    @staticmethod
    def transform(image, mask):
        # Resize
        t1 = Resize((224,224))
        image = t1(image)
        mask = t1(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            t2 = HorizontalFlip()
            image = t2(image)
            mask = t2(mask)

        # Random vertical flipping
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
        
        # Normalize
        ### Ray - Please do not do this.  ToTensor already does this for you
        #t6 = Normalize()
        #image = t6(image)
        #mask = t6(mask)

        return image, mask    

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
              
        ## Removed RGB conversion
        image = Image.open(self.images_path_list[idx])#.convert("RGB")
        mask = Image.open(os.path.splitext(self.images_path_list[idx])[0]+'.png')#.convert("RGB")
    
        image, mask = self.transform(image, mask)

        return image, mask

