import random
import torch
import copy
import re
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

class CNNSubsampling(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 4096,
        kernel_size: int = 5,
        activation_func: str = 'relu',
        norm: str = 'batch',
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        if input_dim * 4 < output_dim:
            self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d1 = nn.Conv1d(input_dim, 2 * input_dim, kernel_size, 1, 0)
            self.bn1 = nn.BatchNorm1d(2 * input_dim, eps=1e-3, momentum=0.99)
            self.relu1 = nn.ReLU()

            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(2 * input_dim, 4 * input_dim, kernel_size, 2, 0)
            self.bn2 = nn.BatchNorm1d(4 * input_dim, eps=1e-3, momentum=0.99)
            self.relu2 = nn.ReLU()
            
            self.project = nn.Linear(4 * input_dim, output_dim)
            self.cnn_num = 2
        else:
            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(input_dim, 2 * input_dim, kernel_size, 2, 0)
            if norm == 'batch':
                self.bn2 = nn.BatchNorm1d(2 * input_dim, eps=1e-3, momentum=0.99)
            elif norm == 'layer':
                self.bn2 = nn.LayerNorm(2 * input_dim, eps=1e-3)
            if activation_func == 'gelu':
                self.relu2 = nn.GELU()
            else:
                self.relu2 = nn.ReLU()
            
            self.project = nn.Linear(2 * input_dim, output_dim)
            self.cnn_num = 1
    
    def forward(self, x, mask_pad, cache=None, return_cache=False):
        """
            x: B, T, input_dim
            mask: (B, T) or (B, 1, T)
        """
        x = x.transpose(1, 2)  # B, channels, T

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.cnn_num == 2:
            if cache is None:
                x = self.left_padding1(x)
            else:
                x = torch.cat((cache[1], x), dim=2)
            if cache is not None:
                cache[1] = x[:, :, 1-self.kernel_size:]
            else:
                cache = [None, x[:, :, 1-self.kernel_size:]]
            x = self.conv1d1(x)
            x = self.bn1(x)
            x = self.relu1(x)

        if cache is None or cache[0] is None:
            x = self.left_padding2(x)
        else:
            x = torch.cat((cache[0], x), dim=2)
        if cache is not None:
            cache[0] = x[:, :, 1-self.kernel_size:]
        else:
            cache = [x[:, :, 1-self.kernel_size:]]
        x = self.conv1d2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.bn2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.relu2(x)
        
        x = x.transpose(1, 2)
        x = self.project(x)

        if return_cache:
            return x, mask_pad[:, :, 0::2], cache
        return x, mask_pad[:, :, 0::2]

class LLMAdapter(nn.Module):
    def __init__(
        self, 
        enc_dim: int, 
        llm_dim: int, 
        reduce_factor: int = 4,
        kernel_size: int = 5, 
        activation_func: str = "gelu", 
        norm: str = "layer"
    ):
        """
        Initializes an LLM Adapter with a dynamic number of subsampling stages.
        Each stage reduces the temporal resolution by a factor of 2.
        """
        super().__init__()

        self.num_blocks = int(math.log2(reduce_factor))
        if 2**self.num_blocks != reduce_factor:
            raise ValueError(f"reduce_factor must be a power of 2 (e.g., 2, 4, 8), got {reduce_factor}")

        subsamplers = []
        for i in range(self.num_blocks):
            out_channels = llm_dim if (i == self.num_blocks - 1) else enc_dim
            subsamplers.append(
                CNNSubsampling(
                    input_dim=enc_dim,
                    output_dim=out_channels,
                    kernel_size=kernel_size,
                    activation_func=activation_func,
                    norm=norm,
                )
            )
        
        self.subsamplers = nn.ModuleList(subsamplers)
    
    def forward(self, x, mask_pad):
        """
        x: (B, T, enc_dim)
        mask_pad: (B, 1, T) or (B, T)
        """
        for subsampler in self.subsamplers:
            x, mask_pad = subsampler(x, mask_pad)
        return x, mask_pad
    
    def load_pretrained_subsamplers(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        for subsampler in self.subsamplers:
            model_dict = subsampler.state_dict()
            filtered = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(filtered)
            subsampler.load_state_dict(model_dict)