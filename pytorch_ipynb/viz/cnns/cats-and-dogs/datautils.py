import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


class CatsDogsDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, img_dir, transform=None):
    
        self.img_dir = img_dir
        
        self.img_names = [i for i in 
                          os.listdir(img_dir) 
                          if i.endswith('.jpg')]
        
        self.y = []
        for i in self.img_names:
            if i.split('.')[0] == 'cat':
                self.y.append(0)
            else:
                self.y.append(1)
        
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.y)
    




def create_cats_and_dogs_dataloaders(batch_size, data_transforms, train_path, valid_path, test_path):
    train_dataset = CatsDogsDataset(img_dir=train_path, 
                                    transform=data_transforms['train'])

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size,
                              drop_last=True,
                              num_workers=4,
                              shuffle=True)

    valid_dataset = CatsDogsDataset(img_dir=valid_path, 
                                    transform=data_transforms['valid'])

    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=batch_size, 
                              num_workers=4,
                              shuffle=False)

    test_dataset = CatsDogsDataset(img_dir=test_path, 
                                   transform=data_transforms['valid'])

    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             num_workers=4,
                             shuffle=False)

    return train_loader, valid_loader, test_loader


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters:
        ------------
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        
        Returns:
        ------------
        Tensor: Normalized image.

        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
    
def convert_rgb_to_grayscale(im_as_arr):
    """
    Converts RGB image to grayscale
    Expects and returns CHW format.
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

