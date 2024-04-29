from train import train_model
from evaluate import evaluate_model
from datasets import BasicDataset
from EfficientUnet import EfficientUnet
from torch.utils.data import DataLoader
import os

import torch
import logging



if __name__ == '__main__':        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EfficientUnet(n_channels=2, n_classes=6)

    train_dir = 'data/teste/images/'
    mask_dir = 'data/teste/masks'


    dataset = BasicDataset(train_dir, mask_dir, False)
    loader_args = dict(batch_size = 1, num_workers = os.cpu_count(), pin_memory = True)     # Batch size = 1 por causa da janela
    val_loader = DataLoader(dataset, shuffle = False, drop_last=True, **loader_args)

    soma1, soma2 = evaluate_model(model, val_loader, device=device)
    print(soma1, soma2)