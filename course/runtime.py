import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
sys.path.insert(0, str(ROOT / 'vendor' / 'dinomaly'))

import numpy as np
import torch

CATEGORIES = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable',
              'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush',
              'transistor', 'zipper']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = json.loads((ROOT / 'configs' / (cfg['variant'] + '.json')).read_text())
    if set(cfg) != set(expected):
        raise ValueError('Config fields must match the checked-in schema')
    for key in ['variant', 'noise_dropout', 'attention', 'supervision', 'loss']:
        if cfg[key] != expected[key]:
            raise ValueError('Variant mechanism mismatch: ' + key)
    if cfg['encoder'] not in ['base', 'small']:
        raise ValueError('Only planned base/small register encoders are supported')
    for key in ['image_size', 'crop_size', 'batch_size', 'steps', 'warmup_steps',
                'save_every', 'eval_size', 'workers']:
        if type(cfg[key]) is not int or cfg[key] < (0 if key in ['workers', 'warmup_steps'] else 1):
            raise ValueError('Invalid integer: ' + key)
    if not 0 <= cfg['warmup_steps'] < cfg['steps']:
        raise ValueError('warmup_steps must be smaller than steps')
    if cfg['crop_size'] % 14 or cfg['crop_size'] > cfg['image_size']:
        raise ValueError('crop_size must be a multiple of 14 and <= image_size')
    if cfg['eval_size'] < 5 or not 0 < cfg['max_ratio'] <= 1:
        raise ValueError('Invalid evaluation size or top-pixel fraction')
    for key in ['lr', 'final_lr', 'grad_clip']:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('Invalid positive number: ' + key)
    if not np.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('Invalid weight decay')
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, str(temp))
    os.replace(str(temp), str(path))


def source_state():
    paths = [ROOT / 'run.py'] + list((ROOT / 'course').glob('*.py'))
    paths += list((ROOT / 'configs').glob('*.json'))
    paths += [p for p in (ROOT / 'vendor').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                                           stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(git_revision=revision, fingerprint=fingerprint, files=hashes)


def environment():
    import importlib.metadata
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                packages={p: importlib.metadata.version(p) for p in
                          ['torchvision', 'numpy', 'pandas', 'scikit-learn', 'timm']})
