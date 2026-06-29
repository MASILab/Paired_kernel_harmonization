import logging
import torch
from torch.autograd import Variable
from Utils.model.pix2pix.pix2pix_model import Pix2PixModel
from Utils.model.pix2pix.cycle_gan_model import CycleGANModel
from Utils.model.pix2pix.cycle_gan_residual_model import ResidualCycleGANModel
from Utils.misc import to_items
import dotsi


logger = logging.getLogger()


class BackboneSpec:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def generate_model_spec_utils(config):
        backbone = config['model']['backbone']
        if backbone == 'pix2pix':
            return BackboneSpecPix2Pix(config)
        elif backbone == 'cycle_gan':
            return BackboneSpecCycleGAN(config)
        elif backbone == 'cycle_gan_residual':
            return BackboneSpecResidualCycleGAN(config)
        else:
            raise NotImplementedError

    def set_model(self):
        raise NotImplementedError

    def load_model(self, checkpoint_path):
        raise NotImplementedError

    @staticmethod
    def save_model(model_dict, check_point_path):
        logger.info(f'Save model to {check_point_path}')
        model_dict['model'].save_networks_as_torch_dict(
            model_dict['epoch'],
            check_point_path
        )

    @staticmethod
    def get_loss_dict(model):
        loss_dict = model.get_current_losses()

        total_loss = 0
        for _, val in loss_dict.items():
            total_loss += val

        return loss_dict, total_loss


class BackboneSpecPix2Pix(BackboneSpec):
    def __init__(self, config):
        super(BackboneSpecPix2Pix, self).__init__(config)

    def set_model(self):
        train_mode = self.config['mode']
        result_dict = {}

        pix2pix_model_opt = dotsi.Dict(self.config['pix2pix'])
        if train_mode == 'initial':
            result_dict['model'] = Pix2PixModel(pix2pix_model_opt)
            result_dict['model'].setup(pix2pix_model_opt)
            result_dict['start_epoch'] = 0
        else:
            raise NotImplementedError

        return result_dict

    def load_model(self, checkpoint_path):
        pix2pix_model_opt = dotsi.Dict(self.config['pix2pix'])
        model = Pix2PixModel(pix2pix_model_opt)
        model.setup(pix2pix_model_opt)
        logger.info(f'Load model from {checkpoint_path}')
        epoch = model.load_networks_as_torch_dict(
            checkpoint_path
        )
        return {
            'epoch': epoch,
            'model': model,
            'optimizer': None
        }

    @staticmethod
    def run_model_inference(dataloader_data_item, model: Pix2PixModel, device):
        data_input = {
            'A': Variable(dataloader_data_item['input'].float().to(device)),
            'B': Variable(dataloader_data_item['target'].float().to(device))
        }

        model.set_input(data_input)
        model.forward()

        result_dict = {
            'input': data_input['A'],
            'target': data_input['B'],
            'predict': model.fake_B
        }

        return result_dict

    @staticmethod
    def run_model_optimization(dataloader_data_item, model, device):
        data_input = {
            'A': Variable(dataloader_data_item['input'].float().to(device)),
            'B': Variable(dataloader_data_item['target'].float().to(device))
        }
        model.set_input(data_input)
        model.optimize_parameters()
        return BackboneSpec.get_loss_dict(model)


class BackboneSpecCycleGAN(BackboneSpec):
    def __init__(self, config):
        super(BackboneSpecCycleGAN, self).__init__(config)

    def set_model(self):
        train_mode = self.config['mode']
        result_dict = {}

        model_opt = dotsi.Dict(self.config[self.config['model']['backbone']])
        if train_mode == 'initial':
            result_dict['model'] = CycleGANModel(model_opt)
            result_dict['model'].setup(model_opt)
            result_dict['start_epoch'] = 0
        else:
            raise NotImplementedError

        return result_dict

    def load_model(self, checkpoint_path):
        model_opt = dotsi.Dict(self.config[self.config['model']['backbone']])
        model = CycleGANModel(model_opt)
        model.setup(model_opt)
        logger.info(f'Load model from {checkpoint_path}')
        epoch = model.load_networks_as_torch_dict(
            checkpoint_path
        )
        return {
            'epoch': epoch,
            'model': model,
            'optimizer': None
        }

    @staticmethod
    def run_model_inference(dataloader_data_item, model: CycleGANModel, device):
        data_input = {
            'A': Variable(dataloader_data_item['source_real'].float().to(device)),
            'B': Variable(dataloader_data_item['target_real'].float().to(device))
        }

        model.set_input(data_input)
        model.forward()

        return {
            'source_fake': model.fake_B,
            'source_rec': model.rec_A,
            'target_fake': model.fake_A,
            'target_rec': model.rec_B
        }

    @staticmethod
    def run_model_optimization(dataloader_data_item, model, device):
        data_input = {
            'A': Variable(dataloader_data_item['source_real'].float().to(device)),
            'B': Variable(dataloader_data_item['target_real'].float().to(device))
        }

        model.set_input(data_input)
        model.optimize_parameters()
        return BackboneSpec.get_loss_dict(model)


class BackboneSpecResidualCycleGAN(BackboneSpec):
    def __init__(self, config):
        super(BackboneSpecResidualCycleGAN, self).__init__(config)

    def set_model(self):
        train_mode = self.config['mode']
        result_dict = {}

        model_opt = dotsi.Dict(self.config[self.config['model']['backbone']])
        if train_mode == 'initial':
            result_dict['model'] = ResidualCycleGANModel(model_opt)
            result_dict['model'].setup(model_opt)
            result_dict['start_epoch'] = 0
        else:
            raise NotImplementedError

        return result_dict

    def load_model(self, checkpoint_path):
        model_opt = dotsi.Dict(self.config[self.config['model']['backbone']])
        model = ResidualCycleGANModel(model_opt)
        model.setup(model_opt)
        logger.info(f'Load model from {checkpoint_path}')
        epoch = model.load_networks_as_torch_dict(
            checkpoint_path
        )
        return {
            'epoch': epoch,
            'model': model,
            'optimizer': None
        }

    @staticmethod
    def run_model_inference(dataloader_data_item, model: ResidualCycleGANModel, device):
        data_input = {
            'A': Variable(dataloader_data_item['source_real'].float().to(device)),
            'B': Variable(dataloader_data_item['target_real'].float().to(device))
        }

        model.set_input(data_input)
        model.forward()

        return {
            'source_fake': model.fake_B,
            'source_rec': model.rec_A,
            'target_fake': model.fake_A,
            'target_rec': model.rec_B
        }

    @staticmethod
    def run_model_optimization(dataloader_data_item, model, device):
        data_input = {
            'A': Variable(dataloader_data_item['source_real'].float().to(device)),
            'B': Variable(dataloader_data_item['target_real'].float().to(device))
        }

        model.set_input(data_input)
        model.optimize_parameters()
        return BackboneSpec.get_loss_dict(model)