# Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding

Implementation of [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/abs/2106.02795) by Li, Si, Li, Hsieh and Bengio.

## Installation

```bash
pip install learnable_fourier_positional_encoding
```

## Usage

```python
import torch
from learnable_fourier_positional_encoding import LearnableFourierPositionalEncoding

G = 3
M = 17
x = torch.randn((97, G, M))
enc = LearnableFourierPositionalEncoding(G, M, 768, 32, 768, 10)
pex = enc(x)
print(pex.shape)
```
