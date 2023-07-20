import torch
import itertools
from tkinter import *
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import swin_transformer_mutil_scale
from .cal_ssim import SSIM
from util import  util
import numpy as np
import math
import cv2
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','content']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B','outputs']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']


        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_A', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_B', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netG_C = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_B', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_C = networks.define_G(opt.input_nc, opt.output_nc, opt.nuf, opt.Unet, opt.norm,not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_C = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.D_noise_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.MSELoss()

            self.mseloss = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),lr=opt.lr,betas=(opt.beta1,0.999))
            # self.optimizer_D = torch.optim.Adam(
            #     itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_C.parameters()),
            #     lr=0.0004, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.0004, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_A = self.real_A.transforms.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.65, hue=0.3)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    def PSNR(self,img1, img2):
        b, _, _, _ = img1.shape
        self.img1 = np.clip(img1, 0, 255)
        self.img2 = np.clip(img2, 0, 255)
        self.mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if self.mse == 0:
            return 100
        self.PIXEL_MAX = 1
        return 20 * math.log10(self.PIXEL_MAX / math.sqrt(self.mse))
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""


        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.fake_B=self.netG_A(self.fake_B1)
          # G_B(G_A(A))

        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        # fake_D = self.fake_B
        # # fake_D = fake_D.cpu().squeeze().detach().numpy()
        # fake_D = util.tensor2im(fake_D)
        # # fake_D = fake_D.astype(np.uint8)
        # # fake_D = np.transpose(fake_D, (1, 2, 0))
        # # print(fake_D.shape,fake_D)
        #
        # gray_fake_D = fake_D[:, :, 0] * 0.299 + fake_D[:, :, 1] * 0.587 + fake_D[:, :, 1] * 0.114
        # # gray_fake_D=gray_fake_D.cpu().squeeze().detach().numpy()
        #
        # percent_max = sum(sum(gray_fake_D >= 0.6)) / sum(sum(gray_fake_D <= 1.0))
        #
        # max_value = np.percentile(gray_fake_D[:], 95)
        # if percent_max < (100 - 95) / 100.:
        #     scale = 0.8 / max_value
        #     fake_D = fake_D * scale
        #     fake_D = np.minimum(fake_D, 1.0)
        #
        # gray_fake_D = fake_D[:, :, 0] * 0.299 + fake_D[:, :, 1] * 0.587 + fake_D[:, :, 1] * 0.114
        #
        # sub_value = np.percentile(gray_fake_D[:], 5)
        #
        # #
        # fake_D = (fake_D - sub_value) * (1. / (1 - sub_value))
        #
        # imgHSV = cv2.cvtColor(fake_D, cv2.COLOR_RGB2HSV)
        #
        # # print(imgHSV.shape)
        # # imgHSV =torch.Tensor(imgHSV)
        # H, S, V = cv2.split(imgHSV)
        # # print(H.shape,S.shape,V.shape)
        # S = np.power(S, 0.6)
        # # print(S.shape)
        # d = cv2.merge([H, S, V])
        # fake_D = cv2.cvtColor(d, cv2.COLOR_HSV2RGB)
        #
        # # fake_D = fake_D.cpu().squeeze().detach().numpy()
        # fake_D = np.minimum(fake_D, 1.0)

        # self.outputs = np.concatenate([fake_L, fake_D_o, fake_D], axis=1)
        # self.im = fake_D
        #
        # self.im = np.minimum(self.im, 1.0)
        # self.im = np.maximum(self.im, 0.0)

        self.outputs = util.tensor2im(self.fake_B)
        self.outputs = cv2.fastNlMeansDenoisingColored(self.outputs, None, 6, 10, 7, 21)
        # ZXX = (np.transpose(self.outputs, (2, 0, 1)) + 1) / 255.0
        # ZXX = ZXX[np.newaxis, :, :, :]
        # ZXX = torch.Tensor(ZXX).cuda()


        self.rec_A = self.netG_B(self.fake_B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        self.ssim=SSIM().cuda()

        # self.ssim_fakeb = self.ssim(self.rec_B, self.real_B).cpu().squeeze().detach().numpy()
        self.ssim_fakeB = self.ssim(self.fake_B,self.real_B).cpu().squeeze().detach().numpy()
        # self.ssim_outputs = self.ssim(self.outputs,self.real_B).cpu().squeeze().detach().numpy()
        # self.psnr_fakeB = self.PSNR(self.rec_B.data.cpu().numpy() * 255, self.real_B.data.cpu().numpy() * 255)
        self.psnr_fakeB = self.PSNR(self.fake_B.data.cpu().numpy() * 255, self.real_B.data.cpu().numpy() * 255)
        # self.psnr_outputs = self.PSNR(self.outputs.data.cpu().numpy() * 255, self.real_B.data.cpu().numpy() * 255)

        print('ssim_noise=',self.ssim_fakeB)
        print('psnr_fakeB= ', self.psnr_fakeB)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_B = self.D_noise_pool.query(self.D_noise)
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    # def backward_D_C(self):
    #     fake_N = self.D_noise_pool.query(self.D_noise)
    #     self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_B, fake_N)
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (self.criterionIdt(self.idt_A, self.real_B)+self.mseloss(self.fake_B,self.real_B)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (self.criterionIdt(self.idt_B, self.real_A)+self.mseloss(self.fake_A,self.real_A)) * lambda_A * lambda_idt
            # self.idt_C = self.netG_C(self.real_B)
            # self.loss_idt_C = self.criterionIdt(self.idt_C,self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_content=self.mseloss(self.fake_B,self.real_B)

        # self.loss_G_C = self.criterionGAN(self.netD_C(self.D_noise),True)
        # self.loss_G_C = self.criterionGAN(self.netD_C(self.zengq), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # self.loss_cycle_C = self.criterionCycle(self.zengq, self.real_B) * lambda_B
        # combined loss and calculate gradients
        # self.loss_noise = self.mseloss(self.D_noise,self.real_B)
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A+self.loss_idt_B+self.loss_content

        self.loss_G.backward(retain_graph=True)

    # def backward_G2(self):
    #         """Calculate the loss for generators G_A and G_B"""
    #         # lambda_idt = self.opt.lambda_identity
    #         # lambda_A = self.opt.lambda_A
    #         lambda_B = self.opt.lambda_B
    #         # # Identity loss
    #         # if lambda_idt > 0:
    #         #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
    #         #     self.idt_A = self.netG_A(self.real_B)
    #         #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
    #         #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
    #         #     self.idt_B = self.netG_B(self.real_A)
    #         #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    #         # else:
    #         #     self.loss_idt_A = 0
    #         #     self.loss_idt_B = 0
    #
    #         # GAN loss D_A(G_A(A))
    #         # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    #         # # GAN loss D_B(G_B(B))
    #         # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    #
    #         self.loss_G_C = self.criterionGAN(self.netD_C(self.zengq), True)
    #         # Forward cycle loss || G_B(G_A(A)) - A||
    #         # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    #         # # Backward cycle loss || G_A(G_B(B)) - B||
    #         # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
    #         self.loss_cycle_C = self.mseloss(self.zengq, self.real_B) * lambda_B
    #         # combined loss and calculate gradients
    #         self.loss_G2 =self.loss_cycle_C + self.loss_G_C

            # self.loss_G2.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        # self.backward_G2()
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        # self.backward_D_C()
        self.optimizer_D.step()  # update D_A and D_B's weights
        # if(self.psnr>=15):
        #print('ssim=', self.ssim_i , 'psnr=', self.psnr,'G=',self.loss_G)