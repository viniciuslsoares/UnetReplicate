from pathlib import Path

from functions import *
from functions import logger

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PNGReader
from minerva.data.datasets import SimpleDataset

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.fabric import seed_everything


def main(
    finetune_data,  # Where the model is going to be trained
    data_path,      # Root path for the data
    num_epochs,     # Epochs to be trained
    batch_size,     
    repetition,     # Repetition and random seed
    learning_rate,  
    cap,            # Amount of samples to be used. 1.0 means 100%
    freeze,         # Freeze backbone
    logs_path,      # Where to save the logs
    gpus,           # Gpus to run       
    full_save_name=False,   # If want to save as a specific name
    linear=False,   # FC to choose from linear or default
):

    # Set general seed**
    seed_everything(repetition)
    if full_save_name:
        save_name = full_save_name
    else:
        if isinstance(cap, float):
            save_name = (
                f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap*100:.0f}%"
            )
        elif isinstance(cap, int):
            save_name = (
                f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap}_img"
            )
        
    logger.info(f"Saving model {save_name}")
    
    # Transforms
    if finetune_data == "f3" or finetune_data == "f3_N":
        logger.info("Using padding of (256,704)")
        padding = Padding(256, 704)
    elif finetune_data == "seam_ai" or finetune_data == "seam_ai_N":
        logger.info("Using padding of (1008,592)")
        padding = Padding(1008, 592)

    # 100% dos dados
    if cap == 1.0:
        
        logger.info("Using 100% of train data")
        train_dataset = SeismicFullDataset(root=data_path, partition='train', transform=padding)
        data_module = SeismicDataModule(
            root = data_path,
            batch_size=batch_size,
            cap=cap,
            drop_last=False,
            transform=padding,
            test_transform=padding,
            train_dataset = train_dataset,
            val_dataset = None,
            test_dataset = None,
        )
        
    else:
        logger.info(f'Using {cap} samples of train data')
        data_module = SeismicDataModule(
            root = data_path,
            batch_size=batch_size,
            cap=cap,
            drop_last=False,
            transform=padding,
            test_transform=padding,
            train_dataset = None,
            val_dataset = None,
            test_dataset = None,
        )

    # Model

    model = get_model(
        pretrain_data,
        learning_rate,
        freeze,
        repetition,
        import_root_path,
        full_path=import_path, 
        linear=linear
    )

    log_dir = Path(logs_path) / save_name / finetune_data
    ckpt_dir = Path(ckpt_path) / save_name / finetune_data
    csv_logger = CSVLogger(log_dir, name=save_name, version=finetune_data)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, dirpath=ckpt_dir, mode="min", monitor="val_loss"
    )

    trainer = Trainer(
        accelerator="gpu",
        logger=csv_logger,
        callbacks=[ckpt_callback],
        max_epochs=num_epochs,
        strategy="auto",
        devices=gpus,
        check_val_every_n_epoch=2,
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")


if __name__ == "__main__":
    main(
        pretrain_data="imagenet",
        finetune_data="seam_ai",
        data_path="/home/vinicius.soares/asml/datasets/tiff_data/seam_ai",
        num_epochs=50,
        batch_size=8,
        repetition=0,
        learning_rate=0.001,
        cap=1.0,
        freeze=True,
        ckpt_path="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ht_logs/train/9",
        logs_path="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ht_logs/train/9",
        import_root_path='ckpt_ht/pretrain/',
        gpus=[0],
    )
