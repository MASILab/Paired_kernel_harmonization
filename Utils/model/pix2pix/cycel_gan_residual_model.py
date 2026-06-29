import torch
import itertools
from .image_pool import ImagePairPool
from .base_model import BaseModel
from . import networks


class ResidualCycleGANModel(BaseModel):
    """
    Implementation of CycleGAN variant with focus on the residual information. We force the generator to predict the
    residual map instead of the CT value. In addition, we pass both of the residual map and input CT slice into the
    discriminator as suggested in the initial pix2pix model.
    """
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert (opt.input_nc == opt.output_nc)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.loss_names = [
            'D_A', 'D_B', 'D_A_cycle', 'D_B_cycle',
            'G_A', 'G_B', 'G_A_cycle', 'G_B_cycle',
            'idt_A', 'idt_A_inv', 'idt_B', 'idt_B_inv', 'idt_A_cycle', 'idt_B_cycle']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_A_cycle', 'D_B_cycle']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain).to(self.device)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain).to(self.device)

        if self.isTrain:  # define discriminators
            # We follow the pix2pix model to input both the raw data and the predicted data.
            # D_A operates on (A, G_A(A)) pairs
            self.netD_A = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain).to(self.device)
            # D_B operates on (B, G_B(B)) pairs
            self.netD_B = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain).to(self.device)

            # Define the discriminators used in the cycle consistency. By doing so, we are actually making the
            # assumption that the input channel number is the same as the output, which always hold for CT.
            self.netD_A_cycle = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain).to(self.device)
            self.netD_B_cycle = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain).to(self.device)

        if self.isTrain:
            # CycleGAN paper:
            # To reduce model oscillation, we follow Shrivastava et al.’s  strategy
            # and update the discriminator using a history of generated images rather than the ones
            # produced by the latest generators. We keep an image buffer that stores the 50 previously created images.
            # =====
            # Do we need this? We can try a couple of run and determine this.
            # self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_residual_pool = ImagePairPool(opt.pool_size)
            self.fake_B_residual_pool = ImagePairPool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # In the future, we can change this L1 constrain on residual with considerations of the "acceptable"
            # difference level
            # self.criterionIdt = torch.nn.L1Loss()
            self.criterionRes = self._residual_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(
                    self.netD_A.parameters(),
                    self.netD_B.parameters(),
                    self.netD_A_cycle.parameters(),
                    self.netD_B_cycle.parameters()
                ),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    @staticmethod
    def _residual_loss(residual_map):
        loss = torch.nn.L1Loss()(residual_map, torch.zeros_like(residual_map))
        return loss

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # A cycle
        self.residual_fake_B_real_A = self.netG_A(self.real_A)
        self.fake_B = self.real_A + self.residual_fake_B_real_A
        self.residual_rec_A_fake_B = self.netG_B(self.fake_B)
        self.rec_A = self.fake_B + self.residual_rec_A_fake_B

        # B cycle
        self.residual_fake_A_real_B = self.netG_B(self.real_B)
        self.fake_A = self.real_B + self.residual_fake_A_real_B
        self.residual_rec_B_fake_A = self.netG_A(self.fake_A)
        self.rec_B = self.fake_A + self.residual_rec_B_fake_A

    def backward_D_basic(self, netD, real, real_residual, fake, fake_residual):
        # Note: stop backprop to the generator by detaching the output from generators.
        # Real
        real_residual_pair = torch.cat((real, real_residual.detach()), 1)
        pred_real = netD(real_residual_pair)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        fake_residual_pair = torch.cat((fake, fake_residual), 1).detach()
        pred_fake = netD(fake_residual_pair)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A, residual_rec_B_fake_A = self.fake_A_residual_pool.query(self.fake_A, self.residual_rec_B_fake_A)
        self.loss_D_A = self.backward_D_basic(
            self.netD_A,
            self.real_A,
            self.residual_fake_B_real_A,
            fake_A,
            residual_rec_B_fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # fake_A = self.fake_A_pool.query(self.fake_A)
        fake_B, residual_rec_A_fake_B = self.fake_B_residual_pool.query(self.fake_B, self.residual_rec_A_fake_B)
        self.loss_D_B = self.backward_D_basic(
            self.netD_B,
            self.real_B,
            self.residual_fake_A_real_B,
            fake_B,
            residual_rec_A_fake_B)

    def backward_D_A_cycle(self):
        self.loss_D_A_cycle = self.backward_D_basic(
            self.netD_A_cycle,
            self.real_A,
            self.residual_fake_B_real_A,
            self.rec_A,
            -1 * self.residual_rec_A_fake_B)

    def backward_D_B_cycle(self):
        self.loss_D_B_cycle = self.backward_D_basic(
            self.netD_B_cycle,
            self.real_B,
            self.residual_fake_A_real_B,
            self.rec_B,
            -1 * self.residual_rec_B_fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Inter-domain
        self.loss_G_A = self.criterionGAN(
            self.netD_A(torch.cat((self.fake_A, self.residual_rec_B_fake_A), 1)), True)
        self.loss_G_B = self.criterionGAN(
            self.netD_B(torch.cat((self.fake_B, self.residual_rec_A_fake_B), 1)), True)

        # Intra-domain
        self.loss_G_A_cycle = self.criterionGAN(
            self.netD_A_cycle(torch.cat((self.rec_A, -1 * self.residual_rec_A_fake_B), 1)), True)
        self.loss_G_B_cycle = self.criterionGAN(
            self.netD_B_cycle(torch.cat((self.rec_B, -1 * self.residual_rec_B_fake_A), 1)), True)

        # Identity
        self.loss_idt_A = self.criterionRes(self.residual_fake_B_real_A) * self.opt.lambda_A_idt
        self.loss_idt_A_inv = self.criterionRes(self.residual_rec_A_fake_B) * self.opt.lambda_A_idt
        self.loss_idt_B = self.criterionRes(self.residual_fake_A_real_B) * self.opt.lambda_B_idt
        self.loss_idt_B_inv = self.criterionRes(self.residual_rec_B_fake_A) * self.opt.lambda_B_idt

        # Cycle
        self.loss_idt_A_cycle = self.criterionRes(self.residual_fake_B_real_A + self.residual_rec_A_fake_B) * \
                                self.opt.lambda_A_cycle
        self.loss_idt_B_cycle = self.criterionRes(self.residual_fake_A_real_B + self.residual_rec_B_fake_A) * \
                                self.opt.lambda_B_cycle

        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_G_A_cycle + self.loss_G_B_cycle + \
                      self.loss_idt_A + self.loss_idt_A_inv + \
                      self.loss_idt_B + self.loss_idt_B_inv + \
                      self.loss_idt_A_cycle + self.loss_idt_B_cycle

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_cycle, self.netD_B_cycle], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_cycle, self.netD_B_cycle], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_A_cycle()
        self.backward_D_B_cycle()
        self.optimizer_D.step()  # update D_A and D_B's weights