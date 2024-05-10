from os.path import splitext
from PIL import Image
import numpy as np
import tifffile
import torch
import math
import torch.nn.functional as Func
import matplotlib.pyplot as plt



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


def calculate_window_positions(image_width, image_height, window_size, step_size):
    window_positions = []

    # Calculate the number of passes in horizontal and vertical directions
    horizontal_passes = math.floor((image_width - window_size) / step_size) + 1
    vertical_passes = math.floor((image_height - window_size) / step_size) + 1

    # Iterate over each pass
    for i in range(vertical_passes):
        for j in range(horizontal_passes):
            # Calculate the initial position of the window
            start_x = j * step_size
            start_y = i * step_size

            # Calculate the final position of the window
            end_x = start_x + window_size
            end_y = start_y + window_size

            # window_positions.append(((start_x, start_y), (end_x, end_y)))
            window_positions.append((start_x, start_y))

    return window_positions


def reconstruct_image(list_coords, list_images, n_classes, out_height, out_width, device):
    
    image_size = list_images[0].shape[2]
    out_image = torch.zeros((n_classes, out_height, out_width))
    
    for i, (x, y) in enumerate(list_coords):
        temp = Func.softmax(torch.tensor(list_images[i]), dim=0).squeeze(dim=0).to(device) # Pvvment aplicar unsqueeze(dim=0)
        slice = out_image[:, y:y+image_size, x:x+image_size].to(device)
        aux = torch.add(slice, temp)
        out_image[:, y:y+image_size, x:x+image_size] = aux
        
    return out_image.argmax(dim=0).unsqueeze(dim=0)      # Ou dim=1, dependendo do shape

    # Verifiar o shape contanto NCHW ou CHW

def plot_and_save(x1, x1_name, filename, x2=None, x2_name=None, x3=None, x3_name=None ,title='title'):
    plt.plot(x1, label=x1_name)
    if x2 is not None:
        plt.plot(x2, label=x2_name)
    if x3 is not None:
        plt.plot(x3, label=x3_name)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close() 
    

def contar_prefixos(lista):
    contagem = {'xl':0, 'il':0}
    for elemento in lista:
        prefixo = elemento[:2]
        if prefixo in contagem:
            contagem[prefixo] += 1
        else:
            contagem[prefixo] = 1
    return contagem