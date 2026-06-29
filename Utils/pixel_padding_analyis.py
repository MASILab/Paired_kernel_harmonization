import numpy as np
import os
import logging
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm


logger = logging.getLogger()


class PixelPadRegionAnalysis:
    def __init__(self, in_ct_dir, out_mask_dir, n_proc=10):
        self.in_ct_dir = in_ct_dir
        self.out_mask_dir = out_mask_dir
        os.makedirs(self.out_mask_dir, exist_ok=True)

        self.n_proc = n_proc

    @staticmethod
    def get_pad_mask2(in_native_nii, out_mask_nii):
        in_native_obj = nib.load(in_native_nii)
        in_native_img = in_native_obj.get_fdata()

        z_variance_map = np.var(in_native_img[:, :, 10:-11], axis=2)

        slice_pad_region = (z_variance_map == 0).astype(int)

        mask_img = np.zeros(in_native_img.shape, dtype=int)
        for z_idx in range(mask_img.shape[2]):
            mask_img[:, :, z_idx] = slice_pad_region

        out_obj = nib.Nifti1Image(mask_img,
                                  affine=in_native_obj.affine,
                                  header=in_native_obj.header)

        nib.save(out_obj, out_mask_nii)

    def generate_pixel_pad_region_mask(self):
        file_list = os.listdir(self.in_ct_dir)

        Parallel(
            n_jobs=self.n_proc,
            prefer='threads'
        )(delayed(self.get_pad_mask2)(
            os.path.join(self.in_ct_dir, file_name),
            os.path.join(self.out_mask_dir, file_name)
        )
          for file_name in tqdm(file_list, total=len(file_list)))