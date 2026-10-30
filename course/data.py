from pathlib import Path

from .runtime import CATEGORIES
from PIL import Image
from torch.utils.data import Dataset
from dataset import get_data_transforms


class MVTec(Dataset):
    """Explicit filename pairing; official image/mask transforms are preserved."""
    def __init__(self, root, cfg, phase, category=None):
        self.root = Path(root).resolve()
        self.image_transform, self.mask_transform = get_data_transforms(cfg['image_size'], cfg['crop_size'])
        self.samples = []
        for cat in ([category] if category else CATEGORIES):
            directory = self.root / cat / phase
            if not directory.is_dir():
                raise ValueError('Missing split: ' + str(directory))
            types = sorted(p.name for p in directory.iterdir() if p.is_dir())
            if (phase == 'train' and types != ['good']) or 'good' not in types:
                raise ValueError('Expected normal-only train and test/good: ' + str(directory))
            if phase == 'test' and len(types) < 2:
                raise ValueError('Missing anomalous test types: ' + cat)
            for defect in types:
                images = sorted((directory / defect).glob('*.png'))
                if not images:
                    raise ValueError('Empty image directory: ' + str(directory / defect))
                for image in images:
                    mask = None if defect == 'good' else self.root / cat / 'ground_truth' / defect / (image.stem + '_mask.png')
                    if mask is not None and not mask.is_file():
                        raise ValueError('Missing mask: ' + str(mask))
                    self.samples.append((image, mask, cat, defect))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        image, mask, cat, defect = self.samples[index]
        with Image.open(image) as source:
            x = self.image_transform(source.convert('RGB'))
        if mask is None:
            gt = torch.zeros(1, x.shape[-2], x.shape[-1])
        else:
            with Image.open(mask) as source:
                gt = self.mask_transform(source.convert('L'))
        return x, gt, int(mask is not None), image.relative_to(self.root).as_posix()


def manifest(dataset):
    from .runtime import sha256
    return [dict(image=p.relative_to(dataset.root).as_posix(), image_sha256=sha256(p),
                 mask=m.relative_to(dataset.root).as_posix() if m else None,
                 mask_sha256=sha256(m) if m else None, category=c, defect=d)
            for p, m, c, d in dataset.samples]
