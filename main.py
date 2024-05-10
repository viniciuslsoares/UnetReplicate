from train import train_model
from evaluate import evaluate_model
from datasets import BasicDataset
from EfficientUnet import EfficientUnet

import torch
import logging






if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    model = EfficientUnet(n_classes=6)
    
    logging.info(f'Network:\n'
            f'\t{model.n_channels} input channels\n'
            f'\t{model.n_classes} output channels (classes)\n')
    
    model.to(device=device)
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    
    train_dir = 'data/images/train'
    train_mask_dir = 'data/annotations/train'
    val_dir = 'data/images/val'
    val_mask_dir = 'data/annotations/val'
    checkpoint_dir = 'checkpoints/'
    
    train_model(
        model=model,
        device=device,
        dir_img_train=train_dir,
        dir_mask_train=train_mask_dir,
        dir_img_val=val_dir,
        dir_mask_val=val_mask_dir,
        dir_checkpoint=checkpoint_dir,
        save_checkpoint=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_percent=0.1,
        augmentation=True,
        gradient_clipping=1.0,
    )