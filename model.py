import funlib.learn.torch as ft
import torch


class SpineUNet(torch.nn.Module):

    def __init__(self, crop_output=None):
        super(SpineUNet, self).__init__()
        self.unet = ft.models.UNet(
            1,
            16, 5, 
            [(2, 2, 2), (2, 2, 2)],
            constant_upsample=True,
            padding='same')
        self.conv = ft.models.unet.ConvPass(
            16, 3,
            [(1, 1, 1)],
            activation='Sigmoid')

        self.crop_output = crop_output

    def forward(self, x):
        x = self.unet(x)
        x = self.conv(x)
        if self.crop_output:
            x = self.crop(x, self.crop_output)
        return x

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-3] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]
