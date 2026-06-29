import os
import numpy as np
from Utils.utils_backbone_spec import BackboneSpec
from Utils.misc import set_device
from glob import glob
import torch
import torch.utils.data
from tqdm import tqdm
from Utils.model.dataset_pix2pix import PerScanDataset
from joblib import Parallel, delayed
from Utils.pixel_padding_analysis import PixelPadRegionAnalysis
import nibabel as nib


class InferenceUtils:
    def __init__(self, config, checkpoint):
        self.config = config
        self.backbone_spec = BackboneSpec.generate_model_spec_utils(self.config)
        self.model = self.backbone_spec.load_model(checkpoint)['model']
        self.device = set_device(self.config)

    def convert_nii_dir(self, in_nii_dir, out_nii_dir):
        os.makedirs(out_nii_dir, exist_ok=True)

        nii_list = glob(os.path.join(in_nii_dir, '*.nii.gz'))
        print(f'Identify {len(nii_list)} scans (.nii.gz)')

        if self.config['pix2pix']['input_nc'] == 3:
            input_slice_idx = 1
        elif self.config['pix2pix']['input_nc'] == 1:
            input_slice_idx = 0
        else:
            raise NotImplementedError

        print(f'Start to convert kernel. Save result to {out_nii_dir}')
        with torch.no_grad():
            self.model.eval()

            for nii_path in tqdm(nii_list, total=len(nii_list)):
                scan_dataset = PerScanDataset(self.config, nii_path)
                scan_dataset.load_data()
                scan_dataloader = torch.utils.data.DataLoader(
                    scan_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    pin_memory=False,
                    num_workers=self.config['data']['num_workers'],
                    drop_last=False)

                converted_scan_idx_slice_map = {}
                for data in scan_dataloader:
                    inference_data_dict = self.backbone_spec.run_model_inference(
                        data, self.model, self.device)

                    slice_idx_list = data['slice_idx'].data.cpu().numpy().tolist()
                    predict_data = inference_data_dict['predict'].data.cpu().numpy()
                    input_data = inference_data_dict['input'].data.cpu().numpy()
                    for idx, slice_idx in enumerate(slice_idx_list):
                        if self.config['data']['target_type'] == 'slice':
                            converted_scan_idx_slice_map[slice_idx] = predict_data[idx, 0, :, :]
                        elif self.config['data']['target_type'] == 'residue':
                            converted_scan_idx_slice_map[slice_idx] = \
                                input_data[idx, input_slice_idx, :, :] + predict_data[idx, 0, :, :]
                        else:
                            raise NotImplementedError

                nii_file_name = os.path.basename(nii_path)
                out_nii = os.path.join(out_nii_dir, nii_file_name)
                scan_dataset.save_scan(converted_scan_idx_slice_map, out_nii)

    @staticmethod
    def generate_fov_mask(in_nii_dir, out_ppr_mask_dir):
        print(f'Generate FOV mask')
        ppv_generator = PixelPadRegionAnalysis(
            in_ct_dir=in_nii_dir,
            out_mask_dir=out_ppr_mask_dir
        )
        ppv_generator.generate_pixel_pad_region_mask()

    @staticmethod
    def correct_non_fov_region(in_nii_dir, ppr_mask_dir, out_nii_dir, ppr_val=-1024):
        print(f'Correct the non fov region')
        os.makedirs(out_nii_dir, exist_ok=True)

        ct_list = os.listdir(in_nii_dir)

        def process_single_case(ct_file_name):
            in_ct = nib.load(os.path.join(in_nii_dir, ct_file_name))
            in_ppr = nib.load(os.path.join(ppr_mask_dir, ct_file_name))

            in_ct_data = in_ct.get_fdata()
            in_ppr_data = in_ppr.get_fdata()

            out_ct_data = in_ct_data.copy()
            out_ct_data[in_ppr_data == 1] = ppr_val

            out_ct = nib.Nifti1Image(
                out_ct_data,
                header=in_ct.header,
                affine=in_ct.affine)

            out_nii = os.path.join(out_nii_dir, ct_file_name)
            nib.save(out_ct, out_nii)

        Parallel(
            n_jobs=10,
            prefer='threads'
        )(delayed(process_single_case)(ct_file_name)
          for ct_file_name in tqdm(ct_list, total=len(ct_list)))