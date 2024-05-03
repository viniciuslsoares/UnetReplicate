import logging
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import Dice
from torchmetrics.image import TotalVariation
import torch.nn.functional as F
from pathlib import Path
from torchmetrics.classification import MulticlassJaccardIndex


from utils import plot_and_save
import matplotlib.pyplot as plt
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
    dice = Dice().to(device=device)
    total_variation = TotalVariation().to(device=device)
    global_step = 0
    
    IoU = MulticlassJaccardIndex(num_classes=model.n_classes).to(device=device)
    
    ce_loss_list = []
    dice_loss_list = []
    tv_loss_list = []
    epoch_loss_list = []
    iou_list = []
    eval_iou_list = []
    eval_f1_list = []
    eval_loss_list = []
    train_loss_list = []
    
    # 4. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        iou_sum = 0 
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                
                assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                masks_pred = model(images)
                iou_sum += float(IoU(masks_pred, true_masks))
                ce_loss = cross_entropy(masks_pred, true_masks)   
                dice_loss = (1 - dice(
                    F.softmax(masks_pred, dim=1),
                    F.one_hot(true_masks, 6).permute(0, 3, 1, 2)
                    ))
                tv_loss = total_variation(
                    F.softmax(masks_pred, dim=1).float()
                    )
                loss = ce_loss * 0.8 + dice_loss * 0.2
                # loss = CROSS_ENTROPY_WHEIGHT * ce_loss + DICE_WHEIGHT * dice_loss + TOTAL_VARIATION_WHEIGHT * tv_loss
    
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
                
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        val_score = evaluate_model(model, val_loader, device, window = False)
                        scheduler.step(val_score[1])
                        logging.info('IoU score: {:.3f}; F1 score: {:.3f}'.format(val_score[1], val_score[0]))
                        eval_iou_list.append(val_score[1])
                        eval_f1_list.append(val_score[0])
                        model.train()
            logging.info('Epoch finished ! IOU: {:.3f}'.format(iou_sum / len(train_loader)))
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint + '/checkpoint{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
        # print(iou_sum)
        # print(epoch_loss)
        print(val_score[1], val_score[0], val_score[2])
        ce_loss_list.append(float(ce_loss)/len(train_loader))
        dice_loss_list.append(float(dice_loss)/len(train_loader))
        tv_loss_list.append(float(tv_loss)/len(train_loader))
        iou_list.append(float(iou_sum)/len(train_loader))
        epoch_loss_list.append(float(epoch_loss))
        eval_loss_list.append(float(val_score[2]))
        train_loss_list.append(float(epoch_loss)/len(train_loader))

    ce_loss_list = np.array(ce_loss_list)
    dice_loss_list = np.array(dice_loss_list)
    tv_loss_list = np.array(tv_loss_list)
    iou_list = np.array(iou_list)
    epoch_loss_list = np.array(epoch_loss_list)
    eval_iou_list = np.array(eval_iou_list)
    eval_f1_list = np.array(eval_f1_list)
    
    plot_and_save(ce_loss_list, 'Cross Entropy Loss', 'outputs/train_losses.png', x2=dice_loss_list, x2_name='Dice Loss', title='Losses')
    plot_and_save(epoch_loss_list, 'Epoch Loss', 'outputs/epoch_loss.png', title='Epoch Loss')
    plot_and_save(iou_list, 'IoU_train', 'outputs/iou.png', x2=eval_iou_list, x2_name='IoU_eval', title='IoU')
    plot_and_save(eval_f1_list, 'F1_eval', 'outputs/eval.png', title='Eval Score', x2=eval_iou_list, x2_name='IoU_eval')
    plot_and_save(train_loss_list, 'Train Loss', x2=eval_loss_list, x2_name='Eval Loss', title='Train x Eval Loss', filename='outputs/train_eval_loss.png')
    