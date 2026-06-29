import numpy as np
import pandas as pd
import os
import logging
import torch.utils.data as data
import nibabel as nib
from scipy.interpolate import interp1d
import random


logger = logging.getLogger()


class UnpairedKernelDataset(data.Dataset):
    def __init__(self, config, data_mode):
        assert data_mode in ['train', 'valid']  # For validation, we only output the combined png plot for prediction.
        self.config = config
        self.data_mode = data_mode

        # set up the data dir and index file.
        self.data_index_dict = {}
        self.data_size_dict = {}
        self.data_dir_dict = {}
        for domain_tag in ['source', 'target']:
            self.data_index_dict[domain_tag] = pd.read_csv(
                os.path.join(
                    self.config['data']['data_dir'],
                    self.config['data'][f'{data_mode}_{domain_tag}_index_csv']))
            self.data_size_dict[domain_tag] = len(self.data_index_dict[domain_tag].index)
            self.data_dir_dict[domain_tag] = os.path.join(
                self.config['data']['data_dir'],
                self.config['data'][f'{data_mode}_{domain_tag}_data_dir'])

    def __len__(self):
        return max(len(self.data_index_dict['source'].index), len(self.data_index_dict['target'].index))

    def __getitem__(self, index):
        # make sure index is within then range
        index_dict = {
            'source': index % self.data_size_dict['source']
        }

        if self.data_mode == 'train':
            # randomize the index of the second domain to avoid fixed pairs.
            index_dict['target'] = random.randint(0, self.data_size_dict['target'] - 1)
        else:
            index_dict['target'] = index % self.data_size_dict['target']

        res_dict = {}
        for domain in ['source', 'target']:
            record = self.data_index_dict[domain].iloc[index_dict[domain]]
            nii_file_name = record['slice_file_name']

            # Process image data
            image_data = nib.load(
                os.path.join(self.data_dir_dict[domain], nii_file_name)).get_fdata()[:, :, 0]
            clip_range = self.config['data']['clip_range']
            scale_range = self.config['data']['scale_range']
            image_data = np.clip(image_data, clip_range[0], clip_range[1])
            normalizer = interp1d(clip_range, scale_range)
            image_data = normalizer(image_data)

            # Data augmentation (flip + rotation (n * 90))
            if self.data_mode == 'train':
                if self.config['data']['aug_flip']:
                    if random.choice([True, False]):
                        image_data = np.flip(image_data)

                if self.config['data']['aug_rotation']:
                    k_rot90 = random.choice(range(4))
                    image_data = np.rot90(image_data, k=k_rot90)

            # Pack the image data into shape of (C, H, W)
            image_data = np.expand_dims(image_data, axis=0)

            # Add to the final output dict
            image_case_name = nii_file_name.replace('.nii.gz', '')
            res_dict[f'{domain}_case_name'] = image_case_name
            res_dict[f'{domain}_real'] = image_data.copy()

        return res_dict


class PerScanDataset(data.Dataset):
    """
     Load each nii scan individually, and formulate the input data for model inference.
     """
    def __init__(self, config, nii_path):
        self.config = config
        self.nii_path = nii_path
        self.img = nib.load(self.nii_path)
        self.img_data = None

    def load_data(self):
        self.img_data = self.img.get_fdata()

    def save_scan(self, scan_idx_slice_map, out_nii):
        new_scan = np.zeros(self.img_data.shape, dtype=float)
        for slice_idx, slice_data in scan_idx_slice_map.items():
            new_scan[:, :, slice_idx] = slice_data[:, :]

        clip_range = self.config['data']['clip_range']
        scale_range = self.config['data']['scale_range']
        new_scan = np.clip(new_scan, scale_range[0], scale_range[1])

        normalizer = interp1d(scale_range, clip_range)
        new_scan = normalizer(new_scan)

        img_obj = nib.Nifti1Image(new_scan,
                                  affine=self.img.affine,
                                  header=self.img.header)
        nib.save(img_obj, out_nii)

    def __len__(self):
        return self.img_data.shape[2]

    def __getitem__(self, slice_idx):
        nii_file_name = os.path.basename(self.nii_path)
        case_name = nii_file_name.replace('.nii.gz', '')

        clip_range = self.config['data']['clip_range']
        scale_range = self.config['data']['scale_range']

        input_data = np.zeros(
            (self.config['pix2pix']['input_nc'],
             self.config['data']['image_size'],
             self.config['data']['image_size']),
            dtype=float)

        if self.config['pix2pix']['input_nc'] == 3:
            slice_above_idx = slice_idx + 1 if slice_idx < self.img_data.shape[2] - 1 else slice_idx
            slice_bellow_idx = slice_idx - 1 if slice_idx > 0 else slice_idx

            input_data[0, :, :] = self.img_data[:, :, slice_above_idx]
            input_data[1, :, :] = self.img_data[:, :, slice_idx]
            input_data[2, :, :] = self.img_data[:, :, slice_bellow_idx]
        elif self.config['pix2pix']['input_nc'] == 1:
            input_data[0, :, :] = self.img_data[:, :, slice_idx]
        else:
            raise NotImplementedError

        input_data = np.clip(input_data, clip_range[0], clip_range[1])
        normalizer = interp1d(clip_range, scale_range)
        input_data = normalizer(input_data)

        return {
            'pid': case_name,
            'slice_idx': slice_idx,
            'case_name': case_name,
            'input': input_data,
            'target': input_data
        }