import sys
sys.path.append("PerceptualSimilarity\\")
import torch
import numpy as np
from torch import nn
# from kornia import color
import torch.nn.functional as F

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


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


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.reshape(input.size(0), -1)


class StegaStampEncoder(nn.Module):
    def __init__(self, secret_size=100, image_size=400):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(secret_size, int(3/64*(image_size)**2), activation='relu', kernel_initializer='he_normal')
        self.image_size = image_size
        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, inputs):
        secrect, image = inputs
        secrect = secrect - .5
        image = image - .5

        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(-1, 3, int(self.image_size/8), int(self.image_size/8))
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)

        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, image_size=400):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(128*64*64, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image


class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size=100):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(16*16*128, 512, activation='relu'),
            Dense(512, secret_size, activation=None))

    def forward(self, image, aggregation):
        B, TC, H, W = image.shape
        T = TC // 3
        image = image.view(T, 3, H, W)

        image = image - .5
        transformed_image = self.stn(image)
        logits = torch.sigmoid(self.decoder(transformed_image))
        watermark_gt = np.array([1, 0] * 16)
        watermark_gt = np.tile(watermark_gt, (logits.shape[0], 1))
        bitacc = np.mean((watermark_gt) == (logits.detach().cpu().numpy() >= 0.5))

        return bitacc >= 27 / 32

    def detect(self, imgs):
        bitaccs = np.zeros(imgs.shape[0])
        for idx in range(imgs.shape[0]):
            image = imgs[idx]

            image = image - .5
            transformed_image = self.stn(image)

            logits = torch.sigmoid(self.decoder(transformed_image))
            watermark_gt = np.array([1, 0] * 16)
            watermark_gt = np.tile(watermark_gt, (logits.shape[0], 1))
            bitacc = np.mean((watermark_gt) == (logits.detach().cpu().numpy() >= 0.5))  # Attack BA-mean aggregation
            bitaccs[idx] = bitacc

        eps = 1e-6
        p = np.clip(bitaccs, eps, 1 - eps).reshape(-1, 1)
        logit_pos = np.log(p / (1 - p))
        logit_neg = np.zeros_like(logit_pos)

        return np.concatenate([logit_neg, logit_pos], axis=1)

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


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



def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc