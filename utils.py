from os.path import splitext
from PIL import Image
import numpy as np
import tifffile
import torch


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext == ".tif":
        tif_data = tifffile.imread(filename)[:,:,0]
        return tif_data
    elif ext == ".png":
        return np.array(Image.open(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def generate_gradient(shape: tuple[int, int]) -> np.ndarray:              
    
    '''
    Inputs in format (H, W)
    Outputs a gradient from 0 to 1 in both x and y directions
    Channel 0 gradient on W and Channel 1 gradient on H
    '''
    
    xx, yy = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
    gradient = np.stack([xx, yy], axis=-1)
    return gradient


def get_input_data(filename) -> np.ndarray: 
    
    '''
    Load image from filename (tiff, png, npy, pt) and generates the gradient on the size of the image
    Concatenate them and return the result in np.array
    (H, W, C), C = 2
    '''
    
    data = load_image(filename)
    gradient = generate_gradient(data.shape)[:,:,1]
    return np.concatenate([np.expand_dims(data, axis=2), np.expand_dims(gradient, axis=2)], axis=2)