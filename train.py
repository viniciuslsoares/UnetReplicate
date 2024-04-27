import logging
import os
import torch

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import Dice
from torchmetrics.image import TotalVariation
import torch.nn.functional as F
from pathlib import Path


from evaluate import evaluate_model
from datasets import BasicDataset


def train_model(
    model, 
    device,
    dir_img_train: str,
    dir_mask_train: str,  # Move this parameter before dir_val
    dir_img_val: str,
    dir_mask_val: str,
    dir_checkpoint: str,
    save_checkpoint: bool = True,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_cp: bool = True,
    augmentation: bool = False,
    gradient_clipping: float = 1.0,
):
    
    # 1. Dataset
    
    train_set = BasicDataset(data_dir = dir_img_train, mask_dir = dir_mask_train, augmentation = augmentation)
    n_train = len(train_set)
    
    val_set = BasicDataset(data_dir = dir_img_val, mask_dir = dir_mask_val, augmentation = False)
    n_val = len(val_set)
    
    # 2. Dataloaders
    
    loader_args = dict(batch_size = batch_size, num_workers = os.cpu_count(), pin_memory = True)
    train_loader = DataLoader(train_set, shuffle = True, **loader_args)
    
    loader_args = dict(batch_size = 1, num_workers = os.cpu_count(), pin_memory = True)     # Batch size = 1 por causa da janela
    val_loader = DataLoader(val_set, shuffle = False, drop_last=True, **loader_args)
    
    # Initialize logging
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    
    # 3. Loss and optimizer
    
    DICE_WHEIGHT = 0.65
    CROSS_ENTROPY_WHEIGHT = 0.25
    TOTAL_VARIATION_WHEIGHT = 0.1
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 5)
    cross_entropy = torch.nn.CrossEntropyLoss()
    grad_scaler = torch.cuda.amp.GradScaler()
    dice = Dice(num_classes=6)
    total_variation = TotalVariation()
    global_step = 0
    
    # 4. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                
                assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
                    masks_pred = model(images)
                    ce_loss = cross_entropy(masks_pred, true_masks)
                    dice_loss = dice(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, 6).permute(0, 3, 1, 2).float()
                        )
                    tv_loss = total_variation(
                        F.softmax(masks_pred, dim=1).float()
                        )
                    loss = CROSS_ENTROPY_WHEIGHT * ce_loss + DICE_WHEIGHT * dice_loss + TOTAL_VARIATION_WHEIGHT * tv_loss
                    
    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Round de validação
                
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        val_score = evaluate_model(model, val_loader, device)
                        scheduler.step(val_score)
                        logging.info('Validation score: {}'.format(val_score))
                
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')