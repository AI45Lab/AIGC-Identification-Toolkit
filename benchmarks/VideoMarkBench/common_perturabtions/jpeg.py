import torch
import torchvision.transforms as T
from PIL import Image
import io

class JPEG(torch.nn.Module):
    def __init__(self, quality=75):
        super().__init__()
        self.quality = quality
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1] == 3, "Input must be [1, 3, H, W]"

        x = x.squeeze(0)  # [3, 512, 512]

        image = self.to_pil(x)

        output = io.BytesIO()
        image.save(output, 'JPEG', quality=self.quality)

        output.seek(0)
        compressed_image = Image.open(output)

        x_compressed = self.to_tensor(compressed_image).unsqueeze(0)  # [1, 3, 512, 512]

        return x_compressed