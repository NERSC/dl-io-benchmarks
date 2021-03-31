import math
import torch
import torch.nn as nn

def down_conv(in_channels, out_channels, kernel_size, batchnorm):
  layers = []

  # equation worked out to produce half the spatial size of input with stride=2
  padding = math.ceil(kernel_size/2.) - 1
  layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding))

  if batchnorm == True:
    layers.append(nn.BatchNorm2d(out_channels))

  layers.append(nn.LeakyReLU(inplace=True))

  return nn.Sequential(*layers)

def up_conv(in_channels, out_channels, kernel_size, batchnorm):
  layers = []

  # equation worked out to produce double the spatial size of input with stride=2
  padding = math.ceil(kernel_size/2.) - 1
  output_padding = 1 if bool(kernel_size%2) else 0
  layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                   stride=2, padding=padding, output_padding=output_padding))

  if batchnorm == True:
    layers.append(nn.BatchNorm2d(out_channels))

  layers.append(nn.ReLU(inplace=True))

  return nn.Sequential(*layers)


class UNet(nn.Module):
  def __init__(self, N_out_channels=4, width_factor=32, kernel_size=4, batchnorm=True):
    super().__init__()
    self.conv_down1 = down_conv(4, 4*width_factor, kernel_size, batchnorm)
    self.conv_down2 = down_conv(4*width_factor, 8*width_factor, kernel_size, batchnorm)
    self.conv_down3 = down_conv(8*width_factor, 16*width_factor, kernel_size, batchnorm)
    self.conv_down4 = down_conv(16*width_factor, 32*width_factor, kernel_size, batchnorm)
    self.conv_down5 = down_conv(32*width_factor, 32*width_factor, kernel_size, batchnorm)
    self.conv_down6 = down_conv(32*width_factor, 32*width_factor, kernel_size, batchnorm)
    self.conv_down7 = down_conv(32*width_factor, 32*width_factor, kernel_size, batchnorm)

    self.conv_up7 = up_conv(32*width_factor, 32*width_factor, kernel_size, batchnorm)
    self.conv_up6 = up_conv(32*width_factor+32*width_factor, 32*width_factor, kernel_size, batchnorm)
    self.conv_up5 = up_conv(32*width_factor+32*width_factor, 32*width_factor, kernel_size, batchnorm)
    self.conv_up4 = up_conv(32*width_factor+32*width_factor, 16*width_factor, kernel_size, batchnorm)
    self.conv_up3 = up_conv(16*width_factor+16*width_factor, 8*width_factor, kernel_size, batchnorm)
    self.conv_up2 = up_conv(8*width_factor+8*width_factor, 4*width_factor, kernel_size, batchnorm)

    # equation worked out to produce double the spatial size of input with stride=2
    padding = math.ceil(kernel_size/2.) - 1
    output_padding = 1 if bool(kernel_size%2) else 0
    self.conv_last = nn.ConvTranspose2d(4*width_factor+4*width_factor,
                                        N_out_channels, kernel_size,
                                        stride=2, padding=padding, output_padding=output_padding)
  def forward(self, x):
    conv1 = self.conv_down1(x)
    conv2 = self.conv_down2(conv1)
    conv3 = self.conv_down3(conv2)
    conv4 = self.conv_down4(conv3)
    conv5 = self.conv_down5(conv4)
    conv6 = self.conv_down6(conv5)
    conv7 = self.conv_down7(conv6)

    x = self.conv_up7(conv7) # 512
    x = torch.cat([x, conv6], dim=1)
    x = self.conv_up6(x) # 512
    x = torch.cat([x, conv5], dim=1)
    x = self.conv_up5(x) # 512
    x = torch.cat([x, conv4], dim=1)
    x = self.conv_up4(x) # 256
    x = torch.cat([x, conv3], dim=1)
    x = self.conv_up3(x) # 128
    x = torch.cat([x, conv2], dim=1)
    x = self.conv_up2(x) # 64
    x = torch.cat([x, conv1], dim=1)
    x = self.conv_last(x) # 5
    return nn.Tanh()(x)
