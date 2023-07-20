import torch
import torch.nn as nn
from torch.nn import init

import functools
from torch.optim import lr_scheduler

from torchvision import transforms
#from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
#from keras.models import Model
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np
from tkinter import *
from skimage import filters,exposure
import matplotlib.pyplot as plt
from skimage.morphology import disk
from matplotlib.font_manager import FontProperties
import torch.nn.functional as F
#import tensorflow as tf
###############################################################################
# Helper Functions
###############################################################################
from .swin_transformer_mutil_scale import swin_self

class Identity(nn.Module):
    def forward(self, x):
        return x




def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，
因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    学习率调整
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    # 自定义调整学习率
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # 等间隔调整学习率

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # 自适应调整学习率
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # 余弦退火调整学习率

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
初始化网络权重。



参数:

net (network)——要初始化的网络

init_type (str)——初始化方法的名称:normal | xavier | kaim |正交
init_gain (float)——法线、xavier和正交的比例因子。
我们在原始的pix2pix和CycleGAN文件中使用“normal”。但xavier和kaim可能会
在某些应用程序中工作得更好。你可以自己试试。
  在深度学习中，神经网络的权重初始化方法对（weight initialization）对模型的收敛速度和性能有着至关重要的影响。说白了，
  神经网络其实就是对权重参数w的不停迭代更新，以期达到较好的性能。在深度神经网络中，随着层数的增多，我们在梯度下降的过程中，
  极易出现梯度消失或者梯度爆炸。因此，对权重w的初始化则显得至关重要，一个好的权重初始化虽然不能完全解决梯度消失和梯度爆炸的问题，
  但是对于处理这两个问题是有很大的帮助的，并且十分有利于模型性能和收敛速度。在这篇博客中，我们主要讨论四种权重初始化方法：

  kaiming提出了一种针对ReLU的初始化方法，一般称作 He initialization。初始化方式为
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    print('初始化网络参数的类型：', init_type)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    print('生成器的初始化norm', norm)
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    我们目前的实现提供了三种类型的鉴别器:
    [basic]:在最初的pix2pix论文中描述的“PatchGAN”分类器。
    可以区分70×70重叠斑块的真假。
    这样的补丁级鉴别器架构具有较少的参数
    比全图像鉴别器和可以工作任意大小的图像
    以完全卷积的方式。
    [n_layers]:在这个模式下，你可以在鉴别器中指定conv层的数量
    使用参数(默认为[basic] (PatchGAN)中使用的3)。
    【pixel】:1x1 PixelGAN鉴别器可以对一个像素进行真假分类。
    它鼓励更大的颜色多样性，但对空间统计没有影响。
    鉴别器已由初始化。对非线性采用漏泄式继电器
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    print('判别器的初始化模型', norm)
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
创建与输入大小相同的标签张量。
参数:
预测(张量)——tpyically从一个鉴别器的预测
target_is_real (bool)——如果ground truth标签用于真实图像或虚假图像

返回:
一个标签张量填满地面真值标签，并与输入的大小
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



# class RelightNet(nn.Module):
#     def __init__(self, channel=4, norm_layer=nn.BatchNorm2d):
#         super(RelightNet, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         model = [nn.ReflectionPad2d(1),
#                  nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=use_bias),
#                  nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(2),
#                   nn.Conv2d(32, 32, kernel_size=5, padding=0, bias=use_bias),
#                   nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(2),
#                   nn.Conv2d(32, 32, kernel_size=5, padding=0, bias=use_bias),
#                   nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(2),
#                   nn.Conv2d(32, 64, kernel_size=5, padding=0, bias=use_bias),
#                   nn.ReLU(True)]
#         model += [
#                   nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2, stride=1, output_padding=0,
#                                      bias=use_bias),
#                   nn.ReLU(True)]
#         model += [
#                   nn.ConvTranspose2d(32, 16, kernel_size=5, padding=2, stride=1,  output_padding=0,
#                                      bias=use_bias),
#                   nn.ReLU(True)]
#         model += [
#                   nn.ConvTranspose2d(16, 3, kernel_size=5, padding=2, stride=1,  output_padding=0,
#                                      bias=use_bias),
#                   nn.ReLU(True)]
#         self.model = nn.Sequential(*model)
#
#         self.inp = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.inp1=nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.out = nn.Sequential(
#             nn.Conv2d(15, 3, 1, 1, 0),
#             nn.ReLU()
#         )
#         self.conv=nn.Sequential(
#             nn.Conv2d(6, 64, 9, 1, 4),
#             nn.ReLU()
#         )
#         self.conv2=nn.Sequential(
#             nn.Conv2d(64, 64,3, 1, 1),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 4, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.conv4= nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
#             nn.ReLU()
#         )
#     def forward(self, input_L,input_R):
#         input_im=torch.cat([input_L,input_R])
#         print(input_im.shape)
#         x1=
#         return R,L
# class DecomNet(nn.Module):
#     def __init__(self, norm_layer=nn.BatchNorm2d):
#         super(DecomNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1,
#                                padding=2)  # nf=56.add padding ,make the data alignment
#         self.prelu1 = nn.PReLU()
#
#         # Shrinking
#         self.conv2 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1, padding=0)
#         self.prelu2 = nn.PReLU()
#
#         # Non-linear Mapping
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.prelu3 = nn.PReLU()
#         self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.prelu4 = nn.PReLU()
#         self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.prelu5 = nn.PReLU()
#         self.conv6 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.prelu6 = nn.PReLU()
#
#         # Expanding
#         self.conv7 = nn.Conv2d(in_channels=12, out_channels=nf, kernel_size=1, stride=1, padding=0)
#         self.prelu7 = nn.PReLU()
#
#         # Deconvolution
#         self.last_part = nn.ConvTranspose2d(in_channels=nf, out_channels=in_nc, kernel_size=9, stride=upscale,
#                                             padding=4, output_padding=3)
#
#     def forward(self, x):  #
#         out = self.prelu1(self.conv1(x))
#         out = self.prelu2(self.conv2(out))
#         out = self.prelu3(self.conv3(out))
#         out = self.prelu4(self.conv4(out))
#         out = self.prelu5(self.conv5(out))
#         out = self.prelu6(self.conv6(out))
#         out = self.prelu7(self.conv7(out))
#         out = self.last_part(out)
#
#         return out


# class ResnetGenerator(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
#
#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """
#
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, n_blocks=6,
#                  padding_type='reflect'):
#         """Construct a Resnet-based generator
#
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#
#         assert (n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         # model = [nn.ConvTranspose2d(ngf , 64,kernel_size=3, stride=2,
#         #                                  padding=1, output_padding=1,
#         #                                  bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
#
#
#         model = [nn.ReflectionPad2d(1),
#                  nn.Conv2d(input_nc, 64, kernel_size=3, padding=0, dilation=1,bias=use_bias),
#                  norm_layer(64),
#                  nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(2),
#                   nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=2, bias=use_bias),
#                   norm_layer(ngf),
#                   nn.ReLU(True)]
#         # model += [nn.Upsample(scale_factor=1),
#         #           norm_layer(64),
#         #           nn.ReLU(True)]
#
#         # n_downsampling = 2
#         # for i in range(n_downsampling):  # add downsampling layers
#         #     mult = 2 ** i
#         #     model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7, stride=2, padding=3, bias=use_bias),
#         #               norm_layer(ngf * mult * 2),
#         #               nn.ReLU(True)]
#         #
#         # mult = 2 ** n_downsampling
#         for i in range(9):  # add ResNet blocks
#
#             model += [ResnetBlock( ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                                   use_bias=use_bias)]
#
#         # for i in range(n_downsampling):  # add upsampling layers
#         #     mult = 2 ** (n_downsampling - i)
#         #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#         #                                  kernel_size=7, stride=2,
#         #                                  padding=3, output_padding=1,
#         #                                  bias=use_bias),
#         #               norm_layer(int(ngf * mult / 2)),
#         #               nn.ReLU(True)]
#         # model += [nn.Upsample(scale_factor=1),
#         #           norm_layer(ngf),
#         #           nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(2),
#                   nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=2, bias=use_bias),
#                   norm_layer(ngf),
#                   nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(1),
#             nn.Conv2d(64, output_nc, kernel_size=3, padding=0, dilation=1, bias=use_bias),
#         ]
#
#         model += [nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#
#
#     def forward(self, input):
#         """Standard forward"""
#         # x1=self.conv1(input)  #300*300*32
#         # x1=self.conv2(input) #300*300*64
#         # x2 = self.conv2(x1)#300*300*3
#         x5 = self.model(input)  # 300*300*3
#         # print(x5.shape)
#         # x6=self.model2(x1)#300*300*64
#         # x7=self.conv4(x6)
#         # # x3=self.model(x7)#300*300*3
#         # x4 = x7+x5
#
#
#         # y1=self.conv2(x4)
#         # y2=self.conv4(y1)
#         # x4=torch.cat((x2, x3), dim=1)#256*256*128
#         # # print(x4.shape)
#         # x5=self.conv4(x4)#300*300*64
#         # print (x4.shape)
#         # y1=torch.cat([x1,x5],dim=1)#256*256*96
#         # print(y1.shape)
#         # y2=self.conv5(y1)#256*256*64
#         # 300*300*3
#         # y3=self.conv6(y2)
#
#         # print(x5.shape)
#
#         # x6=x3+x4
#
#         return x5
#
#
# class ResnetBlock(nn.Module):
#     """Define a Resnet block"""
#
#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         """Initialize the Resnet block
#
#         A resnet block is a conv block with skip connections
#         We construct a conv block with build_conv_block function,
#         and implement skip connections in <forward> function.
#         Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
#         """
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
#
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         """Construct a convolutional block.
#
#         Parameters:
#             dim (int)           -- the number of channels in the conv layer.
#             padding_type (str)  -- the name of padding layer: reflect | replicate | zero
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#             use_bias (bool)     -- if the conv layer uses bias or not
#
#         Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
#         """
#         conv_block = []
#         # p = 0
#         # if padding_type == 'reflect':
#         #     conv_block += [nn.ReflectionPad2d(1)]
#         # elif padding_type == 'replicate':
#         #     conv_block += [nn.ReplicationPad2d(1)]
#         # elif padding_type == 'zero':
#         #     p = 1
#         # else:
#         #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#         conv_block += [nn.Conv2d(dim, 128, kernel_size=3, padding=1, bias=use_bias), norm_layer(128), nn.ReLU(True)]
#
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]
#         # conv_block += [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), nn.ReLU(True)]
#         conv_block += [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), nn.ReLU(True)]
#
#         p = 0
#         # if padding_type == 'reflect':
#         #     conv_block += [nn.ReflectionPad2d(1)]
#         # elif padding_type == 'replicate':
#         #     conv_block += [nn.ReplicationPad2d(1)]
#         # elif padding_type == 'zero':
#         #     p = 1
#         # else:
#         #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         # conv_block += [nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=use_bias), nn.ReLU(True)]
#         conv_block += [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), nn.ReLU(True)]
#         conv_block += [nn.Conv2d(128, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim)]
#         return nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         """Forward function (with skip connections)"""
#         out = x + self.conv_block(x)  # add skip connections
#         return out
# class ResnetGenerator(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
#
#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """
#     # def edge_conv2d(self,im):
#     # # 用nn.Conv2d定义卷积操作
#     #     self.conv_op = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
#     #     # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
#     #     sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
#     #     # 将sobel算子转换为适配卷积操作的卷积核
#     #     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
#     #     # 卷积输出通道，这里我设置为3
#     #     sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
#     #     # 输入图的通道，这里我设置为3
#     #     sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
#     #
#     #     self.conv_op.weight.data = torch.from_numpy(sobel_kernel)
#     #     # print(conv_op.weight.size())
#     #     # print(conv_op, '\n')
#     #
#     #     self.edge_detect = self.conv_op(im)
#     #     # print(torch.max(edge_detect))
#     #
#     #     return self.edge_detect
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, n_blocks=6,
#                  padding_type='reflect'):
#         """Construct a Resnet-based generator
#
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#
#         assert (n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         # model = [nn.ConvTranspose2d(ngf , 64,kernel_size=3, stride=2,
#         #                                  padding=1, output_padding=1,
#         #                                  bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
#
#
#         model = [nn.ReflectionPad2d(1),
#                  nn.Conv2d(3, 64, kernel_size=3, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(2),
#                   nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=2, bias=use_bias),
#                   norm_layer(ngf),
#                   ]
#         for i in range(6):  # add ResNet blocks
#
#             model += [ResnetBlock( ngf,kernel=3,padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                                   use_bias=use_bias)]
#
#         model +=[nn.Tanh()]
#         self.model = nn.Sequential(*model) #w*h*64
#
#
#
#
#         model1 = [nn.ReflectionPad2d(2),
#                   nn.Conv2d(3, 64, kernel_size=3, padding=0, dilation=2, bias=use_bias),
#                   norm_layer(ngf),
#                   nn.ReLU(True)]
#
#         for i in range(6):  # add ResNet blocks
#
#             model1 += [ResnetBlock(ngf,kernel=1,padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                                   use_bias=use_bias)]
#
#         model1 += [
#             nn.Conv2d(64, 64, kernel_size=3, padding=1,dilation=1, bias=use_bias),norm_layer(ngf),
#             nn.Tanh()
#         ]
#         self.model1 = nn.Sequential(*model1)
#
#
#         model2 = [nn.ReflectionPad2d(2),
#                   nn.Conv2d(3, 32, kernel_size=3, padding=0, dilation=2, bias=use_bias),
#                   norm_layer(32),
#                   nn.ReLU(True)]
#
#         for i in range(6):  # add ResNet blocks
#
#             model2 += [
#                 ResnetBlock(32, kernel=3, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                             use_bias=use_bias)]
#
#         model2 += [
#             nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=use_bias), norm_layer(32),
#
#         ]
#         self.model2 = nn.Sequential(*model2)
#
#
#
#
#
#         # model2 = [
#         #           nn.Conv2d(3, 128, kernel_size=3, padding=1,stride=2, dilation=1, bias=use_bias),
#         #           norm_layer(ngf),
#         #           nn.ReLU(True)]
#         # for i in range(3):  # add ResNet blocks
#         #
#         #     model2 += [ResnetBlock(128,kernel=1,padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,
#         #                           use_bias=use_bias)]
#         #
#         # model2 += [
#         #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(ngf),
#         #     nn.Tanh()
#         # ]
#         # self.model2 = nn.Sequential(*model2)
#
#
#
#
#
#
#
#
#         self.conv1_128 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=2, dilation=2),
#             norm_layer(ngf),
#             nn.ReLU(True)
#         )
#         self.conv2_64 = nn.Sequential(
#             nn.Conv2d(64, 3, kernel_size=3, stride=1, dilation=2,padding=2),
#             # ResnetBlock1(3,kernel=1,padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
#             #                       use_bias=use_bias),
#             # norm_layer(3),
#             nn.Tanh()
#         )
#
#
#         self.conv3_96 = nn.Sequential(
#             nn.Conv2d(96, 64, kernel_size=3, stride=1, dilation=2,padding=2),
#             # ResnetBlock1(3,kernel=1,padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
#             #                       use_bias=use_bias),
#             norm_layer(3),
#             nn.Tanh()
#         )
#         self.conv4_192 = nn.Sequential(
#             nn.Conv2d(192, 64, kernel_size=1, stride=1, dilation=1, padding=0),
#             # ResnetBlock1(3,kernel=1,padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
#             #                       use_bias=use_bias),
#
#             nn.Tanh()
#         )
#
#
#     def forward(self, input):
#         """Standard forward"""
#         x1 =self.model(input)#([1, 64, 64, 64])
#         x2 = self.model1(input)  #([1, 64, 64, 64])
#
#
#         x4 = torch.cat([x1,x2],dim=1)#128
#         x4=self.conv1_128(x4)#64
#
#         out1 = self.model2(input)#3
#         #
#         out2 = torch.cat([x4, out1], dim=1)
#         out2 = self.conv3_96(out2)#64
#         out2 =self.conv2_64(out2)
#         # z1 = torch.cat([x4,x1],dim=1)#128
#         # z2 = torch.cat([out2,x2],dim=1)#128
#         # z3 = torch.cat([out1,x2],dim=1)#128
#
#         # z1 = self.conv1_128(z1)#64
#         # z2 = self.conv1_128(z2)#64
#         # z3 = self.conv1_128(z3)#64
#
#
#
#
#         # print(x6.shape)
#
#         return out2
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    # def edge_conv2d(self,im):
    # # 用nn.Conv2d定义卷积操作
    #     self.conv_op = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    #     # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    #     sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    #     # 将sobel算子转换为适配卷积操作的卷积核
    #     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    #     # 卷积输出通道，这里我设置为3
    #     sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    #     # 输入图的通道，这里我设置为3
    #     sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
    #
    #     self.conv_op.weight.data = torch.from_numpy(sobel_kernel)
    #     # print(conv_op.weight.size())
    #     # print(conv_op, '\n')
    #
    #     self.edge_detect = self.conv_op(im)
    #     # print(torch.max(edge_detect))
    #
    #     return self.edge_detect
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # model = [nn.ConvTranspose2d(ngf , 64,kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),norm_layer(ngf),nn.ReLU(True)]


        model = [nn.ReflectionPad2d(2),
                 nn.Conv2d(3, 64, kernel_size=3, padding=0, bias=use_bias,dilation=2),
                 norm_layer(ngf),
                 nn.ReLU()]

        # model += [nn.ReflectionPad2d(1),
        #           nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=1,stride=2, bias=use_bias),
        #           norm_layer(ngf),
        #           nn.ReLU(True)
        #           ]
        for i in range(6):  # add ResNet blocks

            model += [ResnetBlock(ngf,kernel=3,padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias)]

        model += [nn.ReflectionPad2d(2),
                 nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=use_bias, dilation=2),
                 norm_layer(ngf),
                 nn.Tanh()]

        self.model = nn.Sequential(*model) #w*h*64




        model1 = [nn.ReflectionPad2d(0),
                 nn.Conv2d(3, 64, kernel_size=3,stride=2, padding=1, bias=use_bias,dilation=1),
                 norm_layer(ngf),
                 nn.ReLU()]
        for i in range(3):  # add ResNet blocks

            model1 += [ResnetBlock(64,kernel=3,padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias)]
        model1 += [nn.ReflectionPad2d(0),
                   nn.ConvTranspose2d(64, 64, kernel_size=3,stride=2, dilation=1, output_padding=1,padding=1,bias=use_bias),
                   nn.ReLU()
        ]
        self.model1 = nn.Sequential(*model1)


        model2 = [nn.ReflectionPad2d(0),
                 nn.Conv2d(3, ngf, kernel_size=3,stride=2, padding=1, bias=use_bias,dilation=1),
                 norm_layer(ngf),
                 nn.ReLU()]
        model2 += [nn.ReflectionPad2d(0),
                  nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1, bias=use_bias, dilation=1),
                  norm_layer(ngf),
                  nn.ReLU()]
        for i in range(3):  # add ResNet blocks

            model2 += [ResnetBlock(ngf, kernel=3, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                   use_bias=use_bias)]


        model2 += [#nn.ReflectionPad2d(0),
                   # nn.ConvTranspose2d(ngf, 64, kernel_size=3, stride=2, dilation=1, output_padding=1, padding=1,bias=use_bias),
                   nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None),
                   norm_layer(ngf),
                   nn.ReLU()
                   ]
        # model2 += [nn.ReflectionPad2d(2),
        #            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=use_bias, dilation=2),
        #            norm_layer(ngf),
        #            nn.ReLU()
        #            ]

        self.model2 = nn.Sequential(*model2)



        self.conv2_128 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=2,padding=2),
            norm_layer(64),
            nn.ReLU()
        )
        self.conv3_64 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2,padding=2),
            nn.ReLU()
        )
        self.conv5_64 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=5, stride=1, dilation=2, padding=4),
            nn.Tanh()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # 可能应该是nn.Linear(channel, channel)
            nn.Sigmoid()
        )
    def forward(self, input):
        """Standard forward"""



        x1 = self.model(input)#64
        # x2 = self.model1(input)#64
        x3 =self.model2(input)
        # y1 = torch.cat([x2,x3],dim=1)
        # y1 = self.conv2_128(y1)
        # print(x2.shape)
        x_raw = x3
        y = torch.abs(x3)
        y_abs = y
        y = self.gap(y)
        y = torch.flatten(y, 1)
        average = y
        y = self.fc(y)
        y = torch.mul(average, y)
        y = y.unsqueeze(2).unsqueeze(2)
        sub = y_abs - y
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        y = torch.mul(torch.sign(x_raw), n_sub)
        # print(x1.shape,x2.shape)
        # x2 = self.conv5_64(x2)
        # x3 = torch.cat([x1,x2],dim=1)
        # x3 = x1+(0.5*y)
        x4 = torch.cat([y,x1],dim=1)
        x4 = self.conv2_128(x4)
        out = self.conv3_64(x4)
        out = self.conv5_64(out)
        # out = self.conv3_64(out)

        out = out

        # out = torch.cat([out1,x2],dim=1)
        #
        # out = self.conv2_128(out)
        # out1 = self.model2(out)
        # out = self.conv3_64(out)
        # out = self.conv3_64(out)







        # print(x6.shape)

        return out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, kernel,padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim,kernel, padding_type, norm_layer, use_dropout, use_bias)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(

            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1),  # 可能应该是nn.Linear(channel, channel)
            nn.Sigmoid(),
        )
        # self.conv_block2 = self.build_conv_block(dim,)
    def build_conv_block(self, dim, kernel,padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        if kernel==3:
          p = 1
        else:
            p=2
        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     p = 1
        # else:
        #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, 64, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(64), nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # conv_block += [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), nn.ReLU(True)]
        # conv_block += [nn.Conv2d(64, 128, kernel_size=kernel, padding=p, bias=use_bias), nn.ReLU(True)]

        # p = 0
        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     p = 1
        # else:
        #      raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # conv_block += [nn.Conv2d(128, 128, kernel_size=kernel, padding=p, bias=use_bias), nn.ReLU(True)]
        # conv_block += [nn.Conv2d(128, 64, kernel_size=kernel, padding=p, bias=use_bias), nn.ReLU(True)]
        conv_block += [nn.Conv2d(64, dim, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        x_raw = x
        y = torch.abs(x)
        y_abs = y
        y = self.gap(y)
        y = torch.flatten(y, 1)
        average = y
        y = self.fc(y)
        y = torch.mul(average, y)
        y = y.unsqueeze(2).unsqueeze(2)
        sub = y_abs - y
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        y = torch.mul(torch.sign(x_raw), n_sub)
        out = y + self.conv_block(y)  # add skip connections
        return out

class ResnetBlock1(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, kernel,padding_type, norm_layer, use_dropout, use_bias):

        super(ResnetBlock1, self).__init__()
        self.conv_block = self.build_conv_block(dim,kernel, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, kernel,padding_type, norm_layer, use_dropout, use_bias):

        conv_block = []
        if kernel==3:
          p = 1
        else:
            p=0

        conv_block += [nn.Conv2d(dim, 64, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(64), nn.RReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(64, 128, kernel_size=kernel, padding=p, bias=use_bias), nn.RReLU(True)]


        conv_block += [nn.Conv2d(128, 64, kernel_size=kernel, padding=p, bias=use_bias), nn.RReLU(True)]

        conv_block += [nn.Conv2d(64, 3, kernel_size=kernel, padding=p, bias=use_bias)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        x_raw = x
        y = torch.abs(x)

        y_abs = y
        y = self.gap(y)

        y = torch.flatten(y, 1)

        average = y
        y = self.fc(y)

        y = torch.mul(average, y)
        y = y.unsqueeze(2).unsqueeze(2)

        sub = y_abs - y
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        y = torch.mul(torch.sign(x_raw), n_sub)

        out = x + self.conv_block(y)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=2,
                             stride=2, padding=0, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # 输入图的通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    print(torch.max(edge_detect))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),


            nn.Conv2d(ndf * 2, 4*ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),


            nn.Conv2d(4*ndf, 8*ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),

            norm_layer(8 * 2),

            nn.Conv2d(8*ndf,2*ndf,kernel_size=1,stride=1, padding=0, bias=use_bias),

            nn.Conv2d(2 * ndf, 1, kernel_size=3, stride=1, padding=1, bias=use_bias),

            nn.Sigmoid()
        ]


        self.net = nn.Sequential(*self.net)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_nc,  ndf, 1, 1, 0),
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d( ndf, 2 * ndf, 3, 1, padding=2,dilation=2),
        #     nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(2 * ndf, 64, 3, 1, padding=2,dilation=2),
        #     nn.ReLU()
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(2 * ndf, 1, 2, 1, 0),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        """Standard forward."""
        # x1 = self.conv1(input)
        # x2 = self.conv2(x1)
        #
        # # x3=self.conv2(x2)
        # # print (x3.shape)
        # x3 = self.conv3(x2)
        x5 = self.net(input)
        # x6=torch.cat([x3,x5],dim=1)
        # x6=self.conv4(x6)
        return x5
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap