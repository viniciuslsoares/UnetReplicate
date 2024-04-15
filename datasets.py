from utils import *
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from os import listdir
from os.path import isfile, splitext, join
from multiprocessing import Pool
from functools import partial
from pathlib import Path
import logging
import random
from tqdm import tqdm



def unique_mask_values(idx, mask_dir):
    
    '''
    Return the unique values of a mask
    '''

    try:
        mask_file = list(mask_dir.glob(idx + '.*'))[0]        # Use the first sample 
    
    except IndexError:
        raise FileNotFoundError(f'No mask file found for image {idx}')
    
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, data_dir: str, mask_dir: str, transform=None):
        
        # Data and mask directories
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Get the ids of the train data
        self.ids = [splitext(file)[0] for file in listdir(data_dir) if isfile(join(data_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)   
    
    def f_transform(self, image, mask):
        
        # To Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(dim=0)
        
        # # Pad image to H = 256
        image = TF.pad(image, (0, 0, 0, 256 - image.shape[1]), padding_mode='symmetric')
        mask = TF.pad(mask, (0, 0, 0, 256 - mask.shape[1]), padding_mode='symmetric')
        
        # # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random Rotation between -5 and 5 degrees
        angle = random.randint(-5, 5)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        
        return image, mask
    
    @staticmethod
    def preprocess(img, mask_values, is_mask):
        if is_mask:
            mask = np.zeros(img.shape, dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
    
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose(2, 0, 1)
                
            if (img > 1).any():
                img = img / 255
    
        return img
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        data_file = list(self.data_dir.glob(name + '.*'))
        
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(data_file) == 1, f'Either no image or multiple images found for the ID {name}: {data_file}'
        # print(mask_file)
        mask = load_image(mask_file[0])                 # Máscara
        data = get_input_data(data_file[0])             # Dado com o gradiente
            
        mask = self.preprocess(mask, self.mask_values, is_mask=True)
        data = self.preprocess(data, self.mask_values, is_mask=False)
        
        if self.transform:
            data, mask = self.f_transform(data, mask)
        else:
            data = torch.from_numpy(data)
            mask = torch.from_numpy(mask)

        return {
            'image': data,
            'mask': mask
        }