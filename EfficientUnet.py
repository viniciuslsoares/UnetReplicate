from torchvision.models import efficientnet_b0
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from minerva.models.nets.image.unet import _OutConv, _DoubleConv
from minerva.models.nets.base import SimpleSupervisedModel
from typing import Optional


class _Up(nn.Module):
    """
    Módulo de upsampling seguido de convoluções duplas com concatenação.

    Este bloco realiza upsampling (bilinear ou transposed conv) e concatena
    com as features correspondentes da etapa encoder antes de aplicar duas convoluções.

    Args:
        in_channels (int): Número de canais da entrada a ser upsampled.
        cat_channels (int): Número de canais da feature para concatenação.
        out_channels (int): Número de canais de saída após as convoluções.
        bilinear (bool): Se True, usa upsampling bilinear; caso contrário, ConvTranspose2d.
    """

    def __init__(self, in_channels, cat_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )  # in_channels == out_channels
            self.conv = _DoubleConv(
                in_channels + cat_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _DoubleConv(in_channels // 2 + cat_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass do módulo.

        Args:
            x1 (Tensor): Tensor da camada anterior (a ser upsampled).
            x2 (Tensor): Tensor da camada do encoder para concatenação.

        Retorna:
            Tensor: Resultado do upsampling e convoluções após concatenação.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _Up_no_cat(nn.Module):
    """
    Módulo de upsampling seguido de convoluções duplas sem concatenação.

    Args:
        in_channels (int): Número de canais da entrada.
        out_channels (int): Número de canais de saída.
        bilinear (bool): Se True, usa upsampling bilinear; caso contrário, ConvTranspose2d.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _DoubleConv(in_channels // 2, out_channels)

    def forward(self, x):
        """
        Forward pass do módulo.

        Args:
            x (Tensor): Tensor da camada anterior.

        Retorna:
            Tensor: Resultado do upsampling e convoluções.
        """
        x = self.up(x)
        return self.conv(x)


class _EfficientUnet(nn.Module):
    """
    Implementação do EfficientNet-B0 baseada na arquitetura U-Net para segmentação de imagens.

    Usa o backbone EfficientNet-B0 pré-treinado (opcional) como encoder e blocos customizados
    para o decoder. Recebe imagens com `n_channels` canais e produz mapas de segmentação
    com `n_classes` canais.

    Args:
        n_channels (int): Número de canais da imagem de entrada (default=2).
        n_classes (int): Número de classes para segmentação (default=6).
        pretrained_backbone (bool): Se True, usa EfficientNet-B0 pré-treinado (default=False).
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 6,
        pretrained_backbone=False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        )

        net = efficientnet_b0(weights=weights)
        effnetb0_backbone = net.features[:6]

        in_layer = nn.Conv2d(
            n_channels,
            32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

        # Encoder
        self.encoder_block0 = effnetb0_backbone[0]  # (2, 256, 256) -> (32, 128, 128)
        self.encoder_block0[0] = in_layer  # substitui conv inicial para n_channels
        self.encoder_block1 = effnetb0_backbone[1:3]  # (32, 128, 128) -> (24, 64, 64)
        self.encoder_block2 = effnetb0_backbone[3]  # (24, 64, 64) -> (40, 32, 32)
        self.encoder_block3 = effnetb0_backbone[4:6]  # (40, 32, 32) -> (112, 16, 16)

        # Decoder
        self.up1 = _Up(112, 40, 256, bilinear=False)  # (112, 16, 16) -> (256, 32, 32)
        self.up2 = _Up(256, 24, 128, bilinear=False)  # (256, 32, 32) -> (128, 64, 64)
        self.up3 = _Up(128, 32, 64, bilinear=False)  # (128, 64, 64) -> (64, 128, 128)
        self.up4 = _Up_no_cat(64, 32, bilinear=False)  # (64, 128, 128) -> (32, 256, 256)

        # Output
        self.outc = _OutConv(32, n_classes)  # (32, 256, 256) -> (n_classes, 256, 256)

    def forward(self, x):
        """
        Forward pass da rede EfficientUnet.

        Args:
            x (Tensor): Tensor de entrada (batch_size, n_channels, altura, largura).

        Retorna:
            Tensor: Logits de segmentação (batch_size, n_classes, altura, largura).
        """
        x1 = self.encoder_block0(x)
        x2 = self.encoder_block1(x1)
        x3 = self.encoder_block2(x2)
        x4 = self.encoder_block3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)

        logits = self.outc(x)
        return logits


class EfficientUnet(SimpleSupervisedModel):
    """
    Wrapper para treinar e usar o EfficientUnet com o framework SimpleSupervisedModel.

    Args:
        n_channels (int): Número de canais da imagem de entrada (default=2).
        n_classes (int): Número de classes para segmentação (default=6).
        learning_rate (float): Taxa de aprendizado (default=0.001).
        bilinear (bool): Se True, usa upsampling bilinear (não implementado no wrapper atual).
        loss_fn (Optional[nn.Module]): Função de perda. Default é CrossEntropyLoss.
        **kwargs: Argumentos adicionais para o SimpleSupervisedModel.
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 6,
        learning_rate: float = 0.001,
        bilinear: bool = False,
        loss_fn: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__(
            backbone=_EfficientUnet(
                n_channels=n_channels, n_classes=n_classes, pretrained_backbone=False
            ),
            fc=torch.nn.Identity(),
            loss_fn=loss_fn or torch.nn.CrossEntropyLoss(),
            learning_rate=learning_rate,
            flatten=False,
            **kwargs,
        )
