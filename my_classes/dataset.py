from my_classes.transforms import Resize, Rotate, HorizontalFlip, VerticalFlip, Normalize, ToTensor


class SurfaceDefectDetectionDataset(Dataset):
    
    def __init__(self, images_path_list):
        """
        Args:
            images_path_list: list of image files path.
        """
        super(SurfaceDefectDetectionDataset, self).__init__()
        self.images_path_list = images_path_list

    def transform(self, image, mask):
        # Resize
        image = Resize(image, (224,224))
        mask = t (mask, (224,224))

        # Random horizontal flipping
        if random.random() > 0.5:
            image = HorizontalFlip(image)
            mask = HorizontalFlip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = VerticalFlip(image)
            mask = VerticalFlip(mask)

        # Rotate
        angle = random.choice([0, -90, 90, 180])
        image = Rotate(image, angle)
        mask = Rotate (mask, angle)
 
        # Normalize
        image = Normalize(image)
        mask = mask.astype(np.float32)/255
       
        # Transform to tensor
        image = ToTensor(image)
        mask = ToTensor(mask)
        return image, mask    

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
               
        image = Image.open(images_path_list[idx]).convert("RGB")
        mask = Image.open(os.path.splitext(images_path_list[idx])[0]+'.png').convert("RGB")
    
        image, mask = transform(image, mask)

        return image, mask

