import argparse
import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset, get_coco_api_from_dataset
import util.misc as utils

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # In-place unnormalize
    num_channels = tensor.shape[0]
    if num_channels != len(mean):
        mean = mean * (num_channels // len(mean)) + mean[:num_channels % len(mean)]
        std = std * (num_channels // len(std)) + std[:num_channels % len(std)]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main(args):
    utils.init_distributed_mode(args)
    
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
            
    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False
    
    device = torch.device(args.device)
    
    # build model
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    model.eval()
    
    # load weights
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print("Warning: No checkpoint provided!")

    dataset_val = build_dataset(image_set='val', args=args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    num_images = args.num_images
    # Ensure we don't try to sample more images than exist in the dataset
    actual_num_images = min(num_images, len(dataset_val))
    indices = random.sample(range(len(dataset_val)), actual_num_images)
    
    cols = 4
    rows = (actual_num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    if actual_num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for ax_idx, idx in enumerate(indices):
        ax = axes[ax_idx]
        img_tensor, target = dataset_val[idx]
        
        # Inference
        samples = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(samples)
            
        # We scale bounding boxes to target["size"] (which is the size after resizing but before padding).
        # This perfectly matches the unpadded area of img_tensor.
        sizes = torch.stack([target["size"]], dim=0).to(device)
        results = postprocessors['bbox'](outputs, sizes)[0]
        
        # Unnormalize and prepare image for matplotlib
        img_unnorm = unnormalize(img_tensor.clone())
        if img_unnorm.shape[0] > 3:
            # For thyroid dataset with brighness_levels > 3, use the first channel (original image) for visualization
            img = img_unnorm[0].cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
        elif img_unnorm.shape[0] == 1:
            img = img_unnorm[0].cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
        else:
            img = img_unnorm.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')
        
        # Draw Ground Truth
        if "boxes" in target:
            gt_boxes = target["boxes"].cpu().numpy()
            h, w = target["size"]
            for i, box in enumerate(gt_boxes):
                # Target boxes are [cx, cy, bw, bh] normalized
                cx, cy, bw, bh = box
                cx = cx * w
                cy = cy * h
                bw = bw * w
                bh = bh * h
                xmin = cx - bw / 2
                ymin = cy - bh / 2
                rect = patches.Rectangle((xmin, ymin), bw, bh, linewidth=2, edgecolor='g', facecolor='none', label='GT' if i == 0 else "")
                ax.add_patch(rect)
                
        # Draw Predictions
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        boxes = results['boxes'].cpu().numpy() # [xmin, ymin, xmax, ymax]
        
        keep = scores >= args.score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            xmin, ymin, xmax, ymax = box
            bw = xmax - xmin
            bh = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), bw, bh, linewidth=2, edgecolor='r', facecolor='none', label='Pred' if i == 0 else "")
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'Cls:{label} {score:.2f}', color='red', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
            
        # Add legend
        handles, plt_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(plt_labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
    # Hide empty subplots
    for ax_idx in range(actual_num_images, len(axes)):
        axes[ax_idx].axis('off')
            
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'inference_results.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved inference results to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR inference script', parents=[get_args_parser()])
    parser.add_argument('--score_thresh', default=0.3, type=float, help='Confidence threshold for predictions')
    parser.add_argument('--num_images', default=8, type=int, help='Number of images to visualize')
    # Support both spelling variants for robustness
    parser.add_argument('--brightness_levels', type=int, default=None, help='Alias for --brighness_levels')
    args = parser.parse_args()
    
    if args.brightness_levels is not None:
        args.brighness_levels = args.brightness_levels
    
    if not args.output_dir:
        args.output_dir = 'logs/inference'
        
    main(args)
