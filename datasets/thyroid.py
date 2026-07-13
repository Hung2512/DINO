# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Face dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Face_utils.py
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Thyroid dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Thyroid_utils.py
"""
from pathlib import Path

import numpy as np

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as Thyroid_mask

import datasets.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os
import pydicom
from scipy import signal
import concurrent.futures

_IMAGE_CACHE = {}

def _process_single_image_for_cache(args):
    path, root, brighness_levels = args
    try:
        org_im = body_cut(get_im_from_dcm(os.path.join(root, path)))
        im_batch = create_imbatch(org_im, brighness_levels)
        return path, (org_im, im_batch)
    except Exception as e:
        print(f"Error caching {path}: {e}")
        return path, None



class ThyroidDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, brighness_levels=5, cache_mode=True):
        super(ThyroidDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.brighness_levels = brighness_levels
        self.prepare = ConvertThyroidPolysToMask(return_masks)
        
        if cache_mode:
            self._preload_images()
            
    def _preload_images(self):
        paths = [self.coco.loadImgs(id)[0]["file_name"] for id in self.ids]
        to_process = [p for p in paths if p not in _IMAGE_CACHE]
        if to_process:
            print(f"Preloading {len(to_process)} images into memory cache using multiprocessing...")
            args_list = [(p, self.root, self.brighness_levels) for p in to_process]
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
                results = list(executor.map(_process_single_image_for_cache, args_list))
            for p, res in results:
                if res is not None:
                    _IMAGE_CACHE[p] = res
            print("Finished preloading images.")
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        if path in _IMAGE_CACHE:
            org_im, im_batch = _IMAGE_CACHE[path]
        else:
            org_im = body_cut(get_im_from_dcm(os.path.join(self.root, path)))
            im_batch = create_imbatch(org_im, self.brighness_levels)
            _IMAGE_CACHE[path] = (org_im, im_batch)
        return gray_to_pil(org_im), im_batch

    def __getitem__(self, idx):
        # Copy from load image function of CocoDetector
        id = self.ids[idx]
        org_img, img = self._load_image(id)
        target = self._load_target(id)
        # img, target = super(ThyroidDetection, self).__getitem__(idx)
        
        # Additional code
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        org_img, target = self.prepare(org_img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

def get_im_from_dcm(path: str):
    im = pydicom.dcmread(path).pixel_array
    if len(im.shape) == 3:
        im = im[0]
    return im

def body_cut(im: np.ndarray):
    return  im[:256, :256] if im.shape[:2] != (512, 512) else im

def gray_to_pil(im: np.ndarray):
    im = Image.fromarray(np.uint16(im))
    return im

def increase_count(im: np.ndarray, factor: int=1):
    # scharr = np.array([[1, 1, 1],

    #                 [1, 2, 1],

    #                 [1, 1, 1]])*factor
    
    # scharr = np.array([[1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1]])*factor
    if factor <= 0:
        return im
    
    scharr = np.array([[1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 4, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]])*factor
    

    im = signal.convolve2d(im, scharr, boundary='symm', mode='same')
    # im = ndimage.gaussian_filter(im.astype(np.float32)*1000*factor, sigma=1, mode='reflect').astype(np.uint16)
    im[im < 0] = 0
    im[im > 255] = 255
    return im

def create_imbatch(im: np.ndarray, brighness_levels: int):
    # im shape HxW
    bright_factors = [2**i for i in range(brighness_levels - 1)]
    bright_factors.insert(0, 0) # Original image
    
    im_batch = np.vstack([increase_count(im, factor) for factor in bright_factors]).reshape(brighness_levels, *im.shape)
    im_batch = im_batch.transpose(1, 2, 0) # transpose to HxWxnum_level
    return im_batch.astype(np.uint8)

def convert_Thyroid_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = Thyroid_mask.frPyObjects(polygons, height, width)
        mask = Thyroid_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertThyroidPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_Thyroid_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to Thyroid api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_thyroid_transforms(image_set: str, brightness_levels: int = 5) -> T.Compose:
    """
    Constructs the data augmentation pipeline for Thyroid SPECT datasets.
    
    Leverages `datasets.transforms` to ensure bounding box coordinates (xyxy) are 
    strictly preserved and scaled synchronously with spatial transformations.
    
    Args:
        image_set: Split of the dataset ('train', 'val', or 'test').
        brightness_levels: Number of channels generated from the Scharr filter.
        
    Returns:
        T.Compose: The composed transformation pipeline.
    """
    # 1. Thiết lập vector Normalize mở rộng động theo số lượng kênh đặc trưng
    base_means = [0.485, 0.456, 0.406]
    base_stds = [0.229, 0.224, 0.225]
    
    means = base_means * (brightness_levels // 3) + base_means[:brightness_levels % 3]
    stds = base_stds * (brightness_levels // 3) + base_stds[:brightness_levels % 3]
    
    normalize = T.Compose([
        T.Normalize(means, stds) # Normalize sẽ tự động cast boxes từ xyxy tuyệt đối sang cxcywh [0, 1]
    ])

    if image_set == 'train':
        # 2. Điều chỉnh Scale phù hợp cho ảnh Y tế (SPECT thường < 512)
        scales = [256, 288, 320, 352, 384, 416, 448, 480, 512]
        max_size = 512
        
        # 3. Pipeline Augmentation tiêu chuẩn bảo toàn Bounding Box
        return T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomSelect(
                # Nhánh 1: Chỉ Resize ngẫu nhiên
                T.RandomResize(scales, max_size=max_size),
                # Nhánh 2: Resize -> Crop ngẫu nhiên -> Resize lại (Giúp mô hình học scale-invariance)
                T.Compose([
                    T.RandomResize([384, 400, 416]),
                    T.RandomSizeCrop(256, 512), 
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
        
    if image_set in ['val', 'test']:
        return T.Compose([
            T.ToTensor(),
            T.RandomResize([512], max_size=512),
            normalize,
        ])
        
    raise ValueError(f'Unknown image_set: {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Thyroid path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images" / "train", root / "annotations" / f'train.json'),
        "val": (root / "images" / "val", root / "annotations" / f'val.json'),
        "test": (root / "images" / "test", root / "annotations" / f'test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    cache_mode = getattr(args, 'cache_mode', True)
    dataset = ThyroidDetection(img_folder, ann_file, transforms=make_thyroid_transforms(image_set, args.brighness_levels), return_masks=args.masks, brighness_levels=args.brighness_levels, cache_mode=cache_mode)
    return dataset