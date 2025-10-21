import numpy as np
import torch
from src.infer import slide_infer_on_raster

class Dummy(torch.nn.Module):
    def __init__(self): super().__init__(); self.conv = torch.nn.Conv2d(3,1,1)
    def forward(self, x): return self.conv(x)

def test_slide_infer():
    model = Dummy().eval()
    img = (np.random.rand(3,600,700)*255).astype('uint8')
    prob = slide_infer_on_raster(model, img, tile_size=256, overlap=64, batch_size=2, device='cpu')
    assert prob.shape[1:] == img.shape[1:]
