# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import glob
import numpy as np

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

from torchmetrics import MeanMetric
from datasets.map_eval import MapEvaluator
from datasets.thyroid import get_im_from_dcm, body_cut, _IMAGE_CACHE
from util.postprocessing import cal_uptake, calc_rsi_acc
import concurrent.futures


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    data_path = args.data_path if args and hasattr(args, 'data_path') else 'data/thyroid_data'

    # Build DCM file path cache once (avoid slow per-image glob on nested folders)
    print("Building DCM file cache...", flush=True)
    dcm_cache = {}
    for root, dirs, files in os.walk(data_path):
        for fname in files:
            if fname.endswith('.dcm'):
                full_path = os.path.join(root, fname)
                rel = os.path.relpath(full_path, data_path)
                parts = rel.replace('\\', '/').split('/')
                # new data: images/{split}/PATIENT/diagnostic/file.dcm -> key = PATIENT/diagnostic/file.dcm
                if len(parts) >= 3 and parts[0] == 'images':
                    key = '/'.join(parts[2:])
                    dcm_cache[key] = full_path
                # old data: images/{split}/file.dcm -> key = file.dcm
                elif len(parts) >= 2 and parts[0] == 'images':
                    key = parts[-1]
                    dcm_cache[key] = full_path
    print(f"DCM cache built: {len(dcm_cache)} files", flush=True)

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    
    # Custom evaluation setup for thyroid dataset
    try:
        label2iou_thrs = {id + 1: float(item) for id, item in enumerate(args.iou_thrs_list.split(','))}
    except:
        label2iou_thrs = {1: 0.3, 2: 0.5}
    map_evaluator = MapEvaluator(label2iou_thrs=label2iou_thrs)
    rsi_acc_metric = MeanMetric()

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        # Sorting for calculating RSI accuracy
        for output in results:
            desc_order = output['scores'].argsort(descending=True)
            for k, v in output.items():
                output[k] = v[desc_order]
                
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # Custom Thyroid processing block
        def process_thyroid_eval(img_id, result_dict):
            gt_anns = base_ds.loadAnns(base_ds.getAnnIds(img_id))
            img_info = base_ds.loadImgs(img_id)[0]

            ground_truth_dict = {
                'boxes': np.asarray([ann['bbox'] for ann in gt_anns]),
                'labels': np.asarray([ann['category_id'] for ann in gt_anns])
            }
            if len(ground_truth_dict['boxes']) > 0:
                ground_truth_dict['boxes'][:, 2:] += ground_truth_dict['boxes'][:, :2]

            pred_dict = {
                'boxes': result_dict['boxes'].cpu().numpy(),
                'labels': result_dict['labels'].cpu().numpy(),
                'scores': result_dict['scores'].cpu().numpy()
            }
            
            metric_updates = {'ground_truth_dict': ground_truth_dict, 'pred_dict': pred_dict, 'rsi_acc_args': None}
            
            try:
                file_name = img_info["file_name"]
                
                # Check cache first
                img = None
                if file_name in _IMAGE_CACHE:
                    img, _ = _IMAGE_CACHE[file_name]
                else:
                    fp = dcm_cache.get(file_name) or dcm_cache.get(file_name.replace('\\', '/').split('/')[-1])
                    if fp is None:
                        fp_search = glob.glob(f'{data_path}/**/{file_name}', recursive=True)
                        if fp_search:
                            fp = fp_search[0]
                    if fp is not None:
                        img = body_cut(get_im_from_dcm(fp))
                
                if img is not None:
                    if len(set(ground_truth_dict['labels'])) >= 2:
                        gt_lbs = ground_truth_dict['labels']
                        ground_truth_dict['uptakes'] = np.asarray([cal_uptake(img, box) for box in ground_truth_dict['boxes']])
                        
                        gt_rsi_nums = ground_truth_dict['uptakes'][gt_lbs == 2]
                        gt_rsi_dens = ground_truth_dict['uptakes'][gt_lbs == 1]
                        
                        if len(gt_rsi_nums) > 0 and len(gt_rsi_dens) > 0:
                            gt_rsi = gt_rsi_nums[0] / gt_rsi_dens[0]
                            pred_dict['uptakes'] = np.asarray([cal_uptake(img, box) for box in pred_dict['boxes']])
                            metric_updates['rsi_acc_args'] = (pred_dict, gt_rsi, 0.3, 2)
            except Exception as e:
                pass
            return metric_updates

        # Parallel execution for custom metric logic
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(process_thyroid_eval, img_id, result_dict) for img_id, result_dict in res.items()]
            for future in concurrent.futures.as_completed(futures):
                metric_updates = future.result()
                map_evaluator.update(metric_updates['ground_truth_dict'], metric_updates['pred_dict'])
                if metric_updates['rsi_acc_args'] is not None:
                    rsi_acc_metric.update(calc_rsi_acc(*metric_updates['rsi_acc_args']))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            for i, (tgt, res_out, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                _res_bbox = outbbox
                _res_prob = res_out['scores']
                _res_label = res_out['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    try:
        eval_map = map_evaluator.caculate()
        eval_rsi_acc = rsi_acc_metric.compute().item() if rsi_acc_metric.weight > 0 else 0.0
        stats.update(**{
            'thyroid_mAP@.5': eval_map.get(2, 0.0), 
            'shoulder_mAP@.3': eval_map.get(1, 0.0), 
            'overall_mAP': eval_map.get('all', 0.0), 
            'rsi_acc': eval_rsi_acc
        })
        map_evaluator.reset()
        rsi_acc_metric.reset()
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs_res in res.items():
            _scores = outputs_res['scores'].tolist()
            _labels = outputs_res['labels'].tolist()
            _boxes = outputs_res['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res