from train import train_model
from evaluate import evaluate_model
from datasets import BasicDataset
from EfficientUnet import EfficientUnet

import torch
import logging






if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    model = EfficientUnet(n_classes=6)
    
    logging.info(f'Network:\n'
            f'\t{model.n_channels} input channels\n'
            f'\t{model.n_classes} output channels (classes)\n')
    
    model.to(device=device)
    
    
    train_dir = 'data/images/train'
    train_mask_dir = 'data/annotations/train'
    val_dir = 'data/images/val'
    val_mask_dir = 'data/annotations/val'
    checkpoint_dir = 'checkpoints/'
    
    try:
        train_model(
            model=model,
            device=device,
            dir_img_train=train_dir,
            dir_mask_train=train_mask_dir,
            dir_img_val=val_dir,
            dir_mask_val=val_mask_dir,
            dir_checkpoint=checkpoint_dir,
            save_checkpoint=True,
            epochs=5,
            batch_size=1,
            learning_rate=1e-5,
            val_percent=0.1,
            save_cp=True,
            augmentation=False,
            gradient_clipping=1.0,
        )
    
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError!')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            dir_img_train=train_dir,
            dir_mask_train=train_mask_dir,
            dir_img_val=val_dir,
            dir_mask_val=val_mask_dir,
            dir_checkpoint=checkpoint_dir,
            save_checkpoint=True,
            epochs=5,
            batch_size=1,
            learning_rate=1e-5,
            val_percent=0.1,
            save_cp=True,
            augmentation=False,
            gradient_clipping=1.0,
        )
    pass