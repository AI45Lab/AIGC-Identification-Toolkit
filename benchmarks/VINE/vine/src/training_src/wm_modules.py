from torch import nn
import torch

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation='relu'),
            Conv2D(8, 16, 3, strides=2, activation='relu'),
            Conv2D(16, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 1, 3, activation=None))

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output, x