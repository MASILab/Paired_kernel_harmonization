import numpy as np
import pandas as pd
import os
import logging
import torch.utils.data as data
import nibabel as nib
from scipy.interpolate import interp1d
# import Utils.utils_emphysema
import random


logger = logging.getLogger()


class PairKernelDataLoader(data.Dataset):
    def __init__(self, config, data_mode):
        self.config = config
        self.data_mode = data_mode
        data_index_csv = os.path.join(self.config['data']['data_dir'], 'slice_index.csv')
        data_index_df = pd.read_csv(data_index_csv)

        self.mode_slice_df = data_index_df.loc[data_index_df['split'] == data_mode]

        # For test
        # if self.data_mode == 'train':
        #     self.mode_slice_df = self.mode_slice_df.sample(frac=0.01)
        # elif self.data_mode == 'valid':
        #     self.mode_slice_df = self.mode_slice_df.sample(frac=0.1)
        # else:
        #     raise NotImplementedError

        if self.data_mode == 'train':
            self.mode_slice_df = self.mode_slice_df.sample(frac=1.)

    def __len__(self):
        return len(self.mode_slice_df.index)

    def __getitem__(self, index):
        slice_record = self.mode_slice_df.iloc[index]

        in_kernel = self.config['data']['in_kernel']
        out_kernel = self.config['data']['out_kernel']

        in_slice_dir = os.path.join(self.config['data']['data_dir'], in_kernel)
        out_slice_dir = os.path.join(self.config['data']['data_dir'], out_kernel)

        pid = slice_record['pid']
        slice_idx = slice_record['slice_idx']
        slice_file_name = f'{pid}_{slice_idx:03d}.nii.gz'

        slice_dict = {
            'slice_in': nib.load(
                os.path.join(in_slice_dir, slice_file_name)).get_fdata()[:, :, 0],
            'slice_target': nib.load(
                os.path.join(out_slice_dir, slice_record['slice_file_name'])).get_fdata()[:, :, 0]
        }

        # Get the slice above and bellow this slice
        if self.config['pix2pix']['input_nc'] == 3:
            slice_total = slice_record['slice_total']
            slice_above_idx = slice_idx + 1 if slice_idx < slice_total - 1 else slice_idx
            slice_bellow_idx = slice_idx - 1 if slice_idx > 0 else slice_idx
            slice_above_file_name = f'{pid}_{slice_above_idx:03d}.nii.gz'
            slice_bellow_file_name = f'{pid}_{slice_bellow_idx:03d}.nii.gz'

            slice_dict.update({
                'slice_above': nib.load(
                    os.path.join(in_slice_dir, slice_above_file_name)).get_fdata()[:, :, 0],
                'slice_bellow': nib.load(
                    os.path.join(in_slice_dir, slice_bellow_file_name)).get_fdata()[:, :, 0]
            })

        # Process the slice data
        # 1. Clip
        # 2. Rescale
        # 3. Get the residue map in the rescaled space
        clip_range = self.config['data']['clip_range']
        scale_range = self.config['data']['scale_range']
        for key in slice_dict.keys():
            slice_data = slice_dict[key]
            slice_data = np.clip(slice_data, clip_range[0], clip_range[1])
            normalizer = interp1d(clip_range, scale_range)
            slice_dict[key] = normalizer(slice_data)

        if self.config['data']['target_type'] == 'residue':
            slice_dict['residue'] = slice_dict['slice_target'] - slice_dict['slice_in']

        # Pack the data into shape of (C, H, W)
        input_data = np.zeros(
            (self.config['pix2pix']['input_nc'],
             self.config['data']['image_size'],
             self.config['data']['image_size']),
            dtype=float)

        if self.config['pix2pix']['input_nc'] == 3:
            input_data[0, :, :] = slice_dict['slice_above'][:, :]
            input_data[1, :, :] = slice_dict['slice_in'][:, :]
            input_data[2, :, :] = slice_dict['slice_bellow'][:, :]
        elif self.config['pix2pix']['input_nc'] == 1:
            input_data[0, :, :] = slice_dict['slice_in'][:, :]
        else:
            raise NotImplementedError

        target_data = np.zeros(
            (self.config['pix2pix']['output_nc'],
             self.config['data']['image_size'],
             self.config['data']['image_size']),
            dtype=float)

        if self.config['data']['target_type'] == 'slice':
            target_data[0, :, :] = slice_dict['slice_target'][:, :]
        elif self.config['data']['target_type'] == 'residue':
            target_data[0, :, :] = slice_dict['residue'][:, :]
        else:
            raise NotImplementedError

        # Data augmentation, random flipping and rotation in x * 90 degrees.
        if self.config['data']['aug_flip']:
            if random.choice([True, False]):
                for c_idx in range(input_data.shape[0]):
                    input_data[c_idx, :, :] = np.flip(input_data[c_idx, :, :])
                target_data[0, :, :] = np.flip(target_data[0, :, :])

        if self.config['data']['aug_rotation']:
            k_rot90 = random.choice(range(4))
            for c_idx in range(input_data.shape[0]):
                input_data[c_idx, :, :] = np.rot90(input_data[c_idx, :, :], k=k_rot90)
            target_data[0, :, :] = np.rot90(target_data[0, :, :], k=k_rot90)

        res_dict = {
            'pid': str(pid),
            'slice_idx': str(slice_idx),
            'case_name': f'{pid}_{slice_idx:03d}',
            'input': input_data,
            'target': target_data
        }

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
