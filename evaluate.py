import torch
import numpy as np

import torch.nn.functional as Func
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score


from utils import calculate_window_positions, reconstruct_image



def evaluate_model(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    window_size: int = 256,
    step: int = 32,
):
    
    net.eval()
    num_val_batches = len(dataloader)
    
    IoU = MulticlassJaccardIndex(num_classes=net.n_classes)
    F1_score = MulticlassF1Score(num_classes=net.n_classes)
    
    IoU_sum = 0
    F1_sum = 0
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation Round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # Verificar o tamanho se NCHW ou CHW
            # Mudaria o shape[3] e shape[2]
            window_positions = calculate_window_positions(
                image_width=image.shape[3],
                image_height=image.shape[2],
                window_size=window_size,
                step_size=step
            )
            
            image = TF.pad(image, padding=(128, 0, 128, 0), padding_mode='symmetric')   # Faixa de reflexão
            crops = []
            for (x, y) in window_positions:
                crop = image[:, y:y+window_size, x:x+window_size]
                crops.append(crop)
                
            for i, crop in enumerate(crops):
                crops[i] = net(crop)
            
            pred_mask = reconstruct_image(window_positions, crops, net.n_classes, image.shape[2], image.shape[3])
            
            IoU_sum += IoU(pred_mask, mask_true)
            F1_sum += F1_score(pred_mask, mask_true)
            
    net.train()
    return IoU_sum / max(num_val_batches, 1), F1_sum / max(num_val_batches, 1)