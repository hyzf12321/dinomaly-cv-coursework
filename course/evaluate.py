"""Fixed upstream evaluation_batch protocol, with CPU accumulation and traceable outputs."""
from pathlib import Path
import csv
import hashlib
import json

from . import runtime as rt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import cal_anomaly_maps, get_gaussian_kernel, compute_pro
from .data import MVTec, manifest
from .model import build

METRICS = ['image_auroc', 'pixel_auroc', 'pixel_ap', 'aupro']


def protocol(cfg):
    return dict(version='upstream-mvtec-v1', image_size=cfg['image_size'], crop_size=cfg['crop_size'],
                eval_size=cfg['eval_size'], max_ratio=cfg['max_ratio'],
                feature_map='mean cosine distances; bilinear align_corners=True to crop size',
                resize='map bilinear align_corners=False; mask nearest',
                smoothing='5x5 Gaussian sigma=4 after resize; zero padding',
                mask='upstream PIL bilinear resize then center crop; >0 after nearest resize',
                aupro='upstream compute_pro, 200 thresholds, FPR<0.3; normalize by sampled max FPR')


def maps_and_scores(en, de, gt, cfg):
    maps, _ = cal_anomaly_maps(en, de, cfg['crop_size'])
    maps = F.interpolate(maps, size=cfg['eval_size'], mode='bilinear', align_corners=False)
    gt = F.interpolate(gt, size=cfg['eval_size'], mode='nearest').bool()
    kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(maps.device)
    maps = kernel(maps)
    count = max(1, int(maps.shape[-1] * maps.shape[-2] * cfg['max_ratio']))
    scores = torch.sort(maps.flatten(1), dim=1, descending=True)[0][:, :count].mean(dim=1)
    return maps[:, 0], gt[:, 0], scores


def metrics(masks, maps, labels, scores):
    if not np.isfinite(maps).all() or not np.isfinite(scores).all():
        raise ValueError('Non-finite anomaly predictions')
    result = dict(image_auroc=float(roc_auc_score(labels, scores)),
                  pixel_auroc=float(roc_auc_score(masks.ravel(), maps.ravel())),
                  pixel_ap=float(average_precision_score(masks.ravel(), maps.ravel())),
                  aupro=float(compute_pro(masks, maps)))
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError('Undefined metric; inspect predictions and masks instead of filling a value')
    return result


def evaluate(args):
    checkpoint = Path(args.checkpoint).resolve()
    saved = torch.load(str(checkpoint), map_location='cpu')
    cfg = saved['config']
    if saved['step'] != cfg['steps'] and not args.allow_incomplete:
        raise ValueError('Checkpoint is not the fixed endpoint; short probes require --allow-incomplete')
    if saved['provenance']['source']['fingerprint'] != rt.source_state()['fingerprint']:
        raise ValueError('Code/config source differs from training; use the matching source snapshot')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = build(cfg, random_encoder=True)
    model.load_state_dict(saved['model'], strict=True)
    model.to(args.device).eval()
    rows, image_rows, test_manifests = [], [], []
    for category in rt.CATEGORIES:
        dataset = MVTec(args.data, cfg, 'test', category)
        test_manifests.extend(manifest(dataset))
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'])
        all_maps, all_masks, all_labels, all_scores, all_paths = [], [], [], [], []
        with torch.no_grad():
            for images, gt, labels, paths in loader:
                en, de = model(images.to(args.device))
                maps, masks, scores = maps_and_scores(en, de, gt, cfg)
                all_maps.append(maps.cpu().numpy())
                all_masks.append(masks.cpu().numpy().astype(np.uint8))
                all_labels.append(labels.numpy())
                all_scores.append(scores.cpu().numpy())
                all_paths.extend(paths)
        maps, masks = np.concatenate(all_maps), np.concatenate(all_masks)
        labels, scores = np.concatenate(all_labels), np.concatenate(all_scores)
        row = dict(category=category, **metrics(masks, maps, labels, scores))
        rows.append(row)
        np.savez_compressed(str(out / (category + '.npz')), maps=maps, masks=masks,
                            labels=labels, scores=scores, paths=np.asarray(all_paths))
        for path, label, score, mask in zip(all_paths, labels, scores, masks):
            image_rows.append(dict(path=path, category=category, label=int(label), score=float(score),
                                   defect_area_fraction=float(mask.mean())))
        print(json.dumps(row), flush=True)
    rows.append(dict(category='mean', **{key: float(np.mean([r[key] for r in rows])) for key in METRICS}))
    for filename, values in [('metrics.csv', rows), ('images.csv', image_rows)]:
        with (out / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    rt.write_json(out / 'test-data.json', test_manifests)
    rt.write_json(out / 'evaluation.json', dict(
        kind='real-data-evaluation', config=cfg, step=saved['step'],
        fixed_endpoint=saved['step'] == cfg['steps'], protocol=protocol(cfg),
        checkpoint_sha256=rt.sha256(checkpoint), training=saved['provenance'],
        test_sha256=hashlib.sha256(json.dumps(test_manifests, sort_keys=True).encode()).hexdigest(),
        metrics_sha256=rt.sha256(out / 'metrics.csv'), environment=rt.environment(), results=rows))
