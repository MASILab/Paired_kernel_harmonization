from Utils.misc import set_device
from torch.utils.tensorboard import SummaryWriter
from Utils.misc import AverageMeterSet
from Utils.misc import AnalyzeTrainingCurve
from tqdm import tqdm
from scipy.interpolate import interp1d
import numpy as np
import torch
import os
from Utils.misc import to_items
import dotsi
from Utils.utils_backbone_spec import BackboneSpec
from Utils.model.dataset_pix2pix import PairKernelDataLoader
from Utils.model.dataset_cycle_gan import UnpairedKernelDataset
import torch.utils.data
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors


logger = logging.getLogger()


class TrainUtilsBase:
    def __init__(self, config, save_path):
        self.writer = SummaryWriter(save_path)
        self.config = config
        self.save_path = save_path
        self.backbone_spec = BackboneSpec.generate_model_spec_utils(self.config)

        self.model = None
        set_model_dict = self.backbone_spec.set_model()
        self.model = set_model_dict['model']
        self.start_epoch = set_model_dict['start_epoch']
        self.device = set_device(self.config)
        self.dataloader_dict = self._set_dataloader()

        self.debug = False

    def _set_dataloader(self):
        dataloader_dict = {}
        for mode in ['train', 'valid']:
            # We use the 'valid' set (a small set) to visually check if the result make sense or not.
            dataset = self._set_spec_dataset(mode)
            dataloader_dict[mode] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config['data']['batch_size'] if mode in ['train'] else 1,
                shuffle=(mode == 'train'),
                pin_memory=False,
                num_workers=self.config['data']['num_workers'],
                drop_last=(mode in ['train'])
            )
        return dataloader_dict

    def _set_spec_dataset(self, mode):
        raise NotImplementedError

    def train(self):
        backbone_name = self.config['model']['backbone']
        opt = dotsi.Dict(self.config[backbone_name])
        for epoch in range(opt.epoch_count,
                           opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            self.model.update_learning_rate()  # update learning rates in the beginning of every epoch.
            train_loss_meters = self._train_epoch(epoch)

            for key, val in train_loss_meters.meters.items():
                self.writer.add_scalar(f'Loss/train_{key}_loss', val.avg, epoch)

            # We don't have the validation step for pix2pix, GAN don't need validation
            # for key, val in train_loss_meters.meters.items():
            #     self.writer.add_scalar(f'Loss/valid_{key}_loss', val.avg, epoch)

            self.writer.flush()

            # However, we should periodically check if the model improved over the time.
            if (epoch % self.config['env']['checkpoint_interval'] == 0) | (epoch == (opt.n_epochs + opt.n_epochs_decay)):
                self.backbone_spec.save_model(
                    {
                        'epoch': epoch,
                        'model': self.model
                    },
                    os.path.join(self.save_path, f'checkpoint_{epoch}.tar')
                )
                if epoch == (opt.n_epochs + opt.n_epochs_decay):
                    last_model_w_epoch_path = os.path.join(self.save_path, f'checkpoint_{epoch}.tar')
                    last_model_path = os.path.join(self.save_path, f'checkpoint_last_model.tar')
                    ln_cmd = f'ln -sf {last_model_w_epoch_path} {last_model_path}'
                    os.system(ln_cmd)
                pred_combine_dir = os.path.join(self.save_path, 'test_combined_png', f'epoch_{epoch}')
                logger.info(f'Save combined png to {pred_combine_dir}')
                os.makedirs(pred_combine_dir, exist_ok=True)
                self._inference_w_gt(self.dataloader_dict['valid'], pred_combine_dir)
                if epoch > 2:
                    train_curve_analyzer = AnalyzeTrainingCurve(epoch, self.save_path)
                    train_curve_analyzer.analyze_training_curve()

        self.writer.close()

    def _convert_to_hu(self, normalized_slice):
        scale_range = self.config['data']['scale_range']
        clip_range = self.config['data']['clip_range']

        in_slice = np.clip(normalized_slice, scale_range[0], scale_range[1])
        normalizer = interp1d(scale_range, clip_range)
        hu_slice = normalizer(in_slice)
        hu_slice = np.clip(hu_slice, clip_range[0], clip_range[1])

        return hu_slice

    def _train_epoch(self, epoch):
        meters = AverageMeterSet()
        self.model.train()

        train_dataloader = self.dataloader_dict['train']

        p_bar = tqdm(range(len(train_dataloader)))
        for index, data in enumerate(train_dataloader):
            loss_dict, total_loss = self.backbone_spec.run_model_optimization(
                data,
                self.model,
                self.device
            )

            for key, val in loss_dict.items():
                meters.update(key, val, self.config['data']['batch_size'])
            meters.update('total', total_loss, train_dataloader.batch_size)

            backbone_name = self.config['model']['backbone']
            p_bar.set_description(
                "Train Epoch: {epoch:4}/{total_epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {total_loss:3e}.".format(
                    epoch=epoch + 1,
                    total_epochs=self.config[backbone_name]['n_epochs'] + self.config[backbone_name]['n_epochs_decay'],
                    batch=index + 1,
                    iter=len(train_dataloader),
                    total_loss=meters['total'].avg
                )
            )
            p_bar.update()
        p_bar.close()
        return meters

    def _inference_w_gt(self, inference_dataloader, pred_combine_dir):
        raise NotImplementedError


class TrainUtilsPaired(TrainUtilsBase):
    def __init__(self, config, save_path):
        super(TrainUtilsPaired, self).__init__(config, save_path)

    def _set_spec_dataset(self, mode):
        return PairKernelDataLoader(self.config, mode)

    def _inference_w_gt(self, inference_dataloader, pred_combine_dir):
        """
        Run inference on the unseen data, and plot the converted and residue map.
        :param inference_dataloader:
        :param pred_combine_dir:
        :return:
        """
        with torch.no_grad():
            self.model.eval()
            for index, data in tqdm(enumerate(inference_dataloader), total=len(inference_dataloader)):
                inference_data_dict = self.backbone_spec.run_model_inference(
                    data, self.model, self.device)

                # Herein, we assume the batch size always be 1 for inference.
                for key in inference_data_dict.keys():
                    # inference_data_dict[key] = inference_data_dict[key].data.cpu().numpy().squeeze()
                    inference_data_dict[key] = inference_data_dict[key].data.cpu().numpy()[0, :, :, :]
                    if key in ['target', 'predict']:
                        inference_data_dict[key] = inference_data_dict[key][0, :, :]

                if self.config['pix2pix']['input_nc'] == 3:
                    input_slice_idx = 1
                elif self.config['pix2pix']['input_nc'] == 1:
                    input_slice_idx = 0
                else:
                    raise NotImplementedError

                res_dict = {'input': self._convert_to_hu(inference_data_dict['input'][input_slice_idx, :, :])}

                if self.config['data']['target_type'] == 'slice':
                    # Assume output is with only 1 channel
                    res_dict['target'] = self._convert_to_hu(inference_data_dict['target'])
                    # Same assumption as above
                    res_dict['predict'] = self._convert_to_hu(inference_data_dict['predict'])
                elif self.config['data']['target_type'] == 'residue': #Residual learning where the difference between the input and output is added to the input to do the style transfer.
                    residue_gt = inference_data_dict['target']
                    res_dict['target'] = inference_data_dict['input'][input_slice_idx, :, :] + residue_gt
                    res_dict['target'] = self._convert_to_hu(res_dict['target'])

                    residue_pred = inference_data_dict['predict']
                    res_dict['predict'] = inference_data_dict['input'][input_slice_idx, :, :] + residue_pred
                    res_dict['predict'] = self._convert_to_hu(res_dict['predict'])
                else:
                    raise NotImplementedError

                # Plot the prediction
                out_png = os.path.join(pred_combine_dir, data['case_name'][0] + '.png')
                self._plot_combined_predict_residue(
                    res_dict,
                    [-50, 90],  # Muscle
                    [-150, 150],  # Just a guess.
                    out_png)

    @staticmethod
    def _plot_combined_predict_residue(res_dict, ct_value_range, residue_range, out_png):
        # for key, item in res_dict.items():
        #     print(key)
        #     print(item.shape)
        res_dict = res_dict.copy()
        for key in res_dict.keys():
            res_dict[key] = np.rot90(res_dict[key])

        fig, ax = plt.subplots()

        gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
        gs.update(wspace=0.1, hspace=0.1)

        ax_dict = {
            'input': plt.subplot(gs[0]),
            'target': plt.subplot(gs[1]),
            'predict': plt.subplot(gs[2]),
            'residue_target_input': plt.subplot(gs[3]),
            'residue_predict_input': plt.subplot(gs[4]),
            'residue_predict_target': plt.subplot(gs[5])
        }
        for key in ax_dict.keys():
            ax_dict[key].axis('off')

        for slice_key in ['input', 'target', 'predict']:
            show_img = np.clip(res_dict[slice_key], ct_value_range[0], ct_value_range[1])
            ax_dict[slice_key].imshow(
                show_img,
                interpolation='none',
                cmap='gray',
                norm=colors.Normalize(vmin=ct_value_range[0], vmax=ct_value_range[1]))

        residue_dict = {
            'residue_target_input': res_dict['target'] - res_dict['input'],
            'residue_predict_input': res_dict['predict'] - res_dict['input'],
            'residue_predict_target': res_dict['predict'] - res_dict['target']
        }
        for slice_key in ['residue_target_input', 'residue_predict_input', 'residue_predict_target']:
            residue_img = np.clip(residue_dict[slice_key], residue_range[0], residue_range[1])
            ax_dict[slice_key].imshow(
                residue_img,
                interpolation='none',
                cmap='gray',
                norm=colors.Normalize(vmin=residue_range[0], vmax=residue_range[1]))

        # print(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


class TrainUtilsUnpaired(TrainUtilsBase):
    def __init__(self, config, save_path):
        super(TrainUtilsUnpaired, self).__init__(config, save_path)

    def _set_spec_dataset(self, mode):
        return UnpairedKernelDataset(self.config, mode)

    def _inference_w_gt(self, inference_dataloader, pred_combine_dir):
        """
        Run inference on the unseen data
        :param inference_dataloader:
        :param pred_combine_dir:
        :return:
        """
        with torch.no_grad():
            self.model.eval()
            for index, data in tqdm(enumerate(inference_dataloader), total=len(inference_dataloader)):
                inference_data_dict = self.backbone_spec.run_model_inference(
                    data, self.model, self.device)

                # Herein, we assume the batch size always be 1 for inference.
                res_dict = {}
                for key in inference_data_dict.keys():
                    res_dict[key] = self._convert_to_hu(inference_data_dict[key].data.cpu().numpy()[0, 0, :, :])

                # Add the input (source real, target real)
                for domain in ['source', 'target']:
                    res_dict[f'{domain}_real'] = self._convert_to_hu(
                        data[f'{domain}_real'].data.cpu().numpy()[0, 0, :, :])

                # Plot the prediction
                for domain in ['source', 'target']:
                    domain_png_dir = os.path.join(pred_combine_dir, domain)
                    os.makedirs(domain_png_dir, exist_ok=True)
                    out_png = os.path.join(
                        domain_png_dir, data[f'{domain}_case_name'][0] + '.png')

                    domain_dict = {}
                    for image_tag in ['real', 'fake', 'rec']:
                        domain_dict[image_tag] = res_dict[f'{domain}_{image_tag}']

                    self._plot_combined_prediction(
                        domain_dict,
                        [-50, 90],  # Muscle
                        [-150, 150],  # Just a guess.
                        out_png)

    @staticmethod
    def _plot_combined_prediction(domain_dict, ct_value_range, residue_range, out_png):
        domain_dict = domain_dict.copy()
        for key in domain_dict.keys():
            domain_dict[key] = np.rot90(domain_dict[key])

        fig, ax = plt.subplots()

        gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
        gs.update(wspace=0.1, hspace=0.1)

        ax_dict = {
            'real': plt.subplot(gs[0]),
            'fake': plt.subplot(gs[1]),
            'rec': plt.subplot(gs[2]),
            # 'blank': plt.subplot(gs[3]),
            'residual_fake_real': plt.subplot(gs[4]),
            'residual_rec_real': plt.subplot(gs[5])
        }

        for key in ax_dict.keys():
            ax_dict[key].axis('off')

        for img_tag in ['real', 'fake', 'rec']:
            show_img = np.clip(domain_dict[img_tag], ct_value_range[0], ct_value_range[1])
            ax_dict[img_tag].imshow(
                show_img,
                interpolation='none',
                cmap='gray',
                norm=colors.Normalize(vmin=ct_value_range[0], vmax=ct_value_range[1]))

        residual_dict = {
            'residual_fake_real': domain_dict['fake'] - domain_dict['real'],
            'residual_rec_real': domain_dict['rec'] - domain_dict['real']
        }
        for residual_tag, residual_data in residual_dict.items():
            residual_data = np.clip(residual_data, residue_range[0], residue_range[1])
            ax_dict[residual_tag].imshow(
                residual_data,
                interpolation='none',
                cmap='gray',
                norm=colors.Normalize(vmin=residue_range[0], vmax=residue_range[1]))

        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()