import os
import logging
from joblib import Parallel, delayed
import numpy as np
import scipy.ndimage as ndimage
import skimage.measure
from torch.utils.data import Dataset
import SimpleITK as sitk
import fill_voids
import skimage.morphology
import torch
from Utils.model.lungmask_resunet import UNet
from tqdm import tqdm
import skimage


logger = logging.getLogger()


def save_file_contents_list(file_name, item_list):
    # print(f'Save list to file {file_name}')
    # print(f'Number items: {len(item_list)}')
    with open(file_name, 'w') as file:
        for item in item_list:
            file.write(item + '\n')


class ProcessLungMask:
    def __init__(self, config):
        self.config = config
        self.in_folder = config['input']['ct_dir']
        # self.file_list = read_file_contents_list(config['input']['file_list'])
        self.file_list = os.listdir(self.in_folder)
        self.lung_mask_dir = os.path.join(config['output']['root_dir'], 'lung_mask')
        os.makedirs(self.lung_mask_dir, exist_ok=True)
        self.lung_mask_failed_txt = os.path.join(config['output']['root_dir'], 'lung_mask.failed')

    def run(self):
        batchsize = 20
        # logger.info(f'Load model')

        file_list = self.file_list
        # n_process_gpu = self.config['model']['n_process_gpu']
        n_process_gpu = 1

        model = get_model(self.config['model']['model_lung_mask'])

        def process_single_case(file_name):

            # try:
            in_nii = os.path.join(self.in_folder, file_name)
            out_nii_mask = os.path.join(self.lung_mask_dir, file_name)

            if os.path.exists(out_nii_mask) & (not self.config['output']['if_overwrite']):
                logger.info(f'Skip. Already processed.')
                return True

            input_image = get_input_image(in_nii)
            # logger.info(f'Infer lungmask')

            result = apply(input_image, model, force_cpu=False, batch_size=batchsize,
                           volume_postprocessing=True, noHU=False)

            result_out = sitk.GetImageFromArray(result)
            result_out.CopyInformation(input_image)
            # logger.info(f'Save result to: {out_nii_mask}')
            sitk.WriteImage(result_out, out_nii_mask)
            return True
            # except:
            #     print(f'Something wrong with {file_name}')
            #     return False

        process_result_list = Parallel(
            n_jobs=n_process_gpu,
            prefer='threads')(delayed(process_single_case)(filename)
                              for idx, filename in tqdm(enumerate(file_list),
                                                        total=len(file_list), desc='Generate lung masks'))

        failed_case_list = [file_list[idx] for idx in range(len(file_list)) if not process_result_list[idx]]

        save_file_contents_list(self.lung_mask_failed_txt, failed_case_list)


def preprocess(img, label=None, resolution=[192, 192]):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)

    imgmtx[imgmtx < -1024] = -1024
    imgmtx[imgmtx > 600] = 600
    cip_xnew = []
    cip_box = []
    cip_mask = []
    for i in range(imgmtx.shape[0]):
        if label is None:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
        else:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0],
                                           height=resolution[1])
            cip_mask.append(m)
        cip_xnew.append(im)
        cip_box.append(box)
    if label is None:
        return np.asarray(cip_xnew), cip_box
    else:
        return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)


def postrocessing(label_image, spare=[]):
    '''some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. All connected components smaller than min_area will be removed
    '''

    # merge small components to neighbours
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        if r.area > origlabels_maxsub[r.max_intensity]:
            origlabels_maxsub[r.max_intensity] = r.area
            region_to_lobemap[r.label] = r.max_intensity

    # for r in tqdm(regions):
    for r in regions:
        if (r.area < origlabels_maxsub[r.max_intensity] or r.max_intensity in spare) and r.area > 2:
            # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n != spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                regions[regionlabels.index(mapto)].max_intensity]:
                origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity] += myarea
            regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[outmask_mapped == spare] = 0

    if outmask_mapped.shape[0] == 1:
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    shape = labelmap.shape
    r = np.any(labelmap, axis=(1, 2))
    c = np.any(labelmap, axis=(0, 2))
    z = np.any(labelmap, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    rmin -= margin if rmin >= margin else rmin
    rmax += margin if rmax <= shape[0] - margin else rmax
    cmin, cmax = np.where(c)[0][[0, -1]]
    cmin -= margin if cmin >= margin else cmin
    cmax += margin if cmax <= shape[1] - margin else cmax
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin -= margin if zmin >= margin else zmin
    zmax += margin if zmax <= shape[2] - margin else zmax

    if rmax - rmin == 0:
        rmax = rmin + 1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])


def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask


def get_input_image(path):
    if os.path.isfile(path):
        # logger.info(f'Read input: {path}')
        input_image = sitk.ReadImage(path)
    else:
        raise NotImplementedError
    return input_image


def crop_and_resize(img, mask=None, width=192, height=192):
    bmask = simple_bodymask(img)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
    if not mask is None:
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox


def simple_bodymask(img):
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128 / np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape) / 128
    return ndimage.zoom(bodymask, real_scaling, order=0)


def apply(image, model=None, force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False):
    if model is None:
        model = get_model('unet', 'R231')

    inimg_raw = sitk.GetArrayFromImage(image)
    directions = np.asarray(image.GetDirection())
    if len(directions) == 9:
        inimg_raw = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
    del image

    if force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logger.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
            batch_size = 1
            device = torch.device('cpu')
    model.to(device)

    if not noHU:
        tvolslices, xnew_box = preprocess(inimg_raw, resolution=[256, 256])
        tvolslices[tvolslices > 600] = 600
        tvolslices = np.divide((tvolslices + 1024), 1624)
    else:
        # support for non HU images. This is just a hack. The models were not trained with this in mind
        tvolslices = skimage.color.rgb2gray(inimg_raw)
        tvolslices = skimage.transform.resize(tvolslices, [256, 256])
        tvolslices = np.asarray([tvolslices * x for x in np.linspace(0.3, 2, 20)])
        tvolslices[tvolslices > 1] = 1
        sanity = [(tvolslices[x] > 0.6).sum() > 25000 for x in range(len(tvolslices))]
        tvolslices = tvolslices[sanity]
    torch_ds_val = LungLabelsDS_inf(tvolslices)
    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1,
                                                 pin_memory=False)

    timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

    with torch.no_grad():
        # for X in tqdm(dataloader_val):
        for X in dataloader_val:
            X = X.float().to(device)
            prediction = model(X)
            pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
            timage_res = np.vstack((timage_res, pls))

    # postprocessing includes removal of small connected components, hole filling and mapping of small components to
    # neighbors
    if volume_postprocessing:
        outmask = postrocessing(timage_res)
    else:
        outmask = timage_res

    if noHU:
        outmask = skimage.transform.resize(outmask[np.argmax((outmask == 1).sum(axis=(1, 2)))], inimg_raw.shape[:2],
                                           order=0, anti_aliasing=False, preserve_range=True)[None, :, :]
    else:
        outmask = np.asarray(
            [reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
            dtype=np.uint8)

    if len(directions) == 9:
        outmask = np.flip(outmask, np.where(directions[[0, 4, 8]][::-1] < 0)[0])

    return outmask.astype(np.uint8)


def get_model(model_path):
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device('cpu'))
    state_dict = torch.load(model_path)
    model = UNet(n_classes=3, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def apply_fused(image, basemodel='LTRCLobes', fillmodel='R231', force_cpu=False, batch_size=20,
                volume_postprocessing=True, noHU=False):
    '''Will apply basemodel and use fillmodel to mitiage false negatives'''
    mdl_r = get_model('unet', fillmodel)
    mdl_l = get_model('unet', basemodel)
    logger.info("Apply: %s" % basemodel)
    res_l = apply(image, mdl_l, force_cpu=force_cpu, batch_size=batch_size, volume_postprocessing=volume_postprocessing,
                  noHU=noHU)
    logger.info("Apply: %s" % fillmodel)
    res_r = apply(image, mdl_r, force_cpu=force_cpu, batch_size=batch_size, volume_postprocessing=volume_postprocessing,
                  noHU=noHU)
    spare_value = res_l.max() + 1
    res_l[np.logical_and(res_l == 0, res_r > 0)] = spare_value
    res_l[res_r == 0] = 0
    logger.info("Fusing results... this may take up to several minutes!")
    return postrocessing(res_l, spare=[spare_value])


class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, None, :, :].astype(np.float)


def reshape_mask(mask, tbox, origsize):
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
    return res