import torch
from torch import nn

from torchlibrosa.stft import STFT
from timm.models.layers import trunc_normal_

from conformer import Conformer

class SSLModel(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.stft = STFT(
            n_fft=512, hop_length=160, win_length=512, window='hann', 
            center=True, pad_mode='reflect', freeze_parameters=False
        )  

        self.inter_conv = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
        )        

        self.patch_embed = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 128, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 512, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 2, 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, 512))

        self.backbone = Conformer(
            depth=8, dim=512, dim_head=64, heads=8, ff_mult=4, 
            conv_expansion_factor=2, conv_kernel_size=31, 
            attn_dropout=0.1, ff_dropout=0.1)

        self.norm = nn.LayerNorm(512)
        self.head = nn.Linear(512, 36)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, sources):
        bsz, n_mic, n_sample = sources.shape

        real, imag = self.stft(sources.reshape(bsz * n_mic, n_sample))
        real = real.reshape(bsz, n_mic, *real.shape[2:])
        imag = imag.reshape(bsz, n_mic, *imag.shape[2:])
        phase = torch.atan2(imag, real)

        x = phase[:, [1]] - phase[:, [0]]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)[..., 1:] # b, 4, T, F

        x = self.inter_conv(x) # b, 8, T, F
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        x = torch.cat((self.cls_token.expand(bsz, -1, -1), x), dim=1)
        x = x + self.pos_embed[:, :(x.shape[1]), :]

        x = self.backbone(x)
        x = self.norm(x)
        return self.head(x[:, 0]), self.head(x[:, 1])