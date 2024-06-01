import lightning as L
import numpy as np
import torch
from minerva.data.datasets.gradient_dataset import GradientDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.models.nets.EfficientUnet import EfficientUnet
from minerva.transforms.transform import _Transform, TransformPipeline
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from matplotlib import pyplot as plt
from pathlib import Path
import os
from matplotlib.colors import ListedColormap

class Padding(_Transform):
    def __init__(self, target_h_size: int, target_w_size: int):
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded = np.expand_dims(padded, axis=2)
            padded = torch.from_numpy(padded).float()
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded = torch.from_numpy(padded).float()

        padded = np.transpose(padded, (2, 0, 1))
        return padded

class SqueezeTarget(_Transform):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 1:
            return np.squeeze(x, axis=0).to(dtype=torch.long)
        else: return x

class F3DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        transforms: _Transform = None,
        batch_size: int = 1,
        num_workers: int = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReader(self.train_path / "train")
            train_label_reader = PNGReader(self.annotations_path / "train")
            train_dataset = GradientDataset(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms,
            )

            val_img_reader = TiffReader(self.train_path / "val")
            val_label_reader = PNGReader(self.annotations_path / "val")
            val_dataset = GradientDataset(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms,
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReader(self.train_path / "test")
            test_label_reader = PNGReader(self.annotations_path / "test")
            test_dataset = GradientDataset(
                readers=[test_img_reader, test_label_reader],
                transforms=self.transforms,
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


label_cmap = ListedColormap(
    [
        [0.29411764705882354, 0.4392156862745098, 0.7333333333333333],
        [0.5882352941176471, 0.7607843137254902, 0.8666666666666667],
        [0.8901960784313725, 0.9647058823529412, 0.9764705882352941],
        [0.9803921568627451, 0.8745098039215686, 0.4666666666666667],
        [0.9607843137254902, 0.47058823529411764, 0.29411764705882354],
        [0.8470588235294118, 0.1568627450980392, 0.1411764705882353],
    ]
)

train_path = "/workspace/f3/images"
annotation_path = "/workspace/f3/annotations"

transform = TransformPipeline([
    Padding(256, 704),
])

data_module = F3DataModule(
    train_path=train_path,
    annotations_path=annotation_path,
    transforms=transform,
    batch_size=1,
)

model = EfficientUnet(
    n_channels=2,
    n_classes=6,
    train_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
    val_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
    test_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
)

trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=[0],
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    save_run_status=True
)

pipeline.run(data=data_module, task="fit")

pipeline.run(data=data_module, task="test")

preds = pipeline.run(data=data_module, task="predict")

image = torch.argmax(preds[31], dim=1)

plt.imsave("prediction.png", image.squeeze().numpy(), cmap=label_cmap)