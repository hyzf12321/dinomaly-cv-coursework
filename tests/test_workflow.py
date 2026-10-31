"""Synthetic/random-weight CPU checks only; never paper experiment results."""
import contextlib
import copy
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from course import runtime as rt
import numpy as np
import torch
from torch import nn
from PIL import Image
from course.model import build, optimizer_scheduler, loss_value, parameters
from course.data import MVTec
from course.train import train, batches
from course import train as training
from course.evaluate import maps_and_scores, metrics
from course.evaluate import evaluate
from course.summarize import summarize
from utils import evaluation_batch

torch.set_num_threads(2)


def small_config(name='F'):
    cfg = rt.config(rt.ROOT / 'configs' / (name + '.json'))
    cfg.update(encoder='small', image_size=56, crop_size=56, batch_size=4,
               steps=10, warmup_steps=2, save_every=1, workers=0, eval_size=16)
    return cfg


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.bottleneck = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Dropout(0.2))
        self.decoder = nn.Conv2d(4, 3, 1)

    def forward(self, x):
        return [x], [self.decoder(self.bottleneck(x))]


class WorkflowTests(unittest.TestCase):
    def test_01_vendor_integrity_and_variants(self):
        source = json.loads((rt.ROOT / 'vendor' / 'SOURCE.json').read_text())
        for name, expected in source['files'].items():
            self.assertEqual(rt.sha256(rt.ROOT / 'vendor' / 'dinomaly' / name), expected, name)
        for name in ['B', 'F', 'A1', 'A2', 'A3']:
            cfg = small_config(name)
            rt.seed_all(1)
            model = build(cfg, random_encoder=True)
            optimizer, scheduler = optimizer_scheduler(model, cfg)
            model.train()
            en, de = model(torch.randn(2, 3, 56, 56))
            self.assertEqual(len(en), 8 if name == 'B' else 2)
            self.assertEqual(model.bottleneck[0].drop.p, cfg['noise_dropout'])
            self.assertEqual(type(model.decoder[0].attn).__name__,
                             'LinearAttention2' if cfg['attention'] == 'linear' else 'Attention')
            before = next(model.decoder.parameters()).detach().clone()
            loss = loss_value(en, de, cfg, 0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            torch.nn.utils.clip_grad_norm_(parameters(model), cfg['grad_clip'])
            optimizer.step()
            scheduler.step()
            # Warmup starts at LR 0; the next update must change parameters.
            optimizer.zero_grad()
            en, de = model(torch.randn(2, 3, 56, 56))
            loss_value(en, de, cfg, 1).backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, next(model.decoder.parameters())))
            del model, optimizer, scheduler, en, de, loss
            gc.collect()

    def test_02_official_evaluation_parity(self):
        cfg = small_config()
        cfg['eval_size'] = 32
        rt.seed_all(3)
        model = ToyModel().eval()
        images = torch.randn(4, 3, 56, 56)
        gt = torch.zeros(4, 1, 56, 56)
        gt[2:, :, 14:28, 14:28] = 1
        labels = torch.tensor([0, 0, 1, 1])
        loader = [(images, gt, labels, ['synthetic'] * 4)]
        with torch.no_grad():
            maps, masks, scores = maps_and_scores(*model(images), gt, cfg)
        actual = metrics(masks.numpy().astype(np.uint8), maps.numpy(), labels.numpy(), scores.numpy())
        # Upstream creates unused CUDA timer events even when evaluating on CPU.
        with patch('torch.cuda.Event', return_value=None):
            official = evaluation_batch(model, loader, 'cpu', max_ratio=cfg['max_ratio'], resize_mask=32)
        for key, index in [('image_auroc', 0), ('pixel_auroc', 3), ('pixel_ap', 4), ('aupro', 6)]:
            self.assertAlmostEqual(actual[key], official[index], places=12)

    def test_03_training_resume_exact_and_export(self):
        with tempfile.TemporaryDirectory(dir=str(rt.ROOT / '.cache')) as tmp:
            root = Path(tmp)
            cfg = small_config()
            cfg_path = root / 'probe.json'
            rt.write_json(cfg_path, cfg)
            data = root / 'data'
            for cat in rt.CATEGORIES:
                directory = data / cat / 'train' / 'good'
                directory.mkdir(parents=True)
                for index in range(2):
                    Image.fromarray(np.random.RandomState(index).randint(0, 255, (56, 56, 3), dtype=np.uint8)).save(directory / ('%03d.png' % index))
            encoder = root / 'encoder.pt'
            encoder.write_bytes(b'synthetic test placeholder; build is mocked')
            args = SimpleNamespace(config=cfg_path, data=data, encoder=encoder, output=root / 'full',
                                   seed=1, device='cpu', resume=None, purpose='probe')
            with patch('course.train.build', side_effect=lambda *a: ToyModel()), contextlib.redirect_stdout(io.StringIO()):
                train(args)
                args.output = root / 'resumed'
                original_save = training.save_checkpoint

                def interrupt(*a, **kw):
                    original_save(*a, **kw)
                    if a[5] == 3:
                        raise InterruptedError('synthetic interruption')

                with patch('course.train.save_checkpoint', side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        train(args)
                args.resume = args.output / 'last.pt'
                train(args)
                full = torch.load(str(root / 'full' / 'last.pt'))
                resumed = torch.load(str(root / 'resumed' / 'last.pt'))
                for key in full['model']:
                    self.assertTrue(torch.equal(full['model'][key], resumed['model'][key]), key)
                for key, value in full['scheduler'].items():
                    other = resumed['scheduler'][key]
                    self.assertTrue(np.array_equal(value, other) if isinstance(value, np.ndarray) else value == other)
                for key, state in full['optimizer']['state'].items():
                    for field, value in state.items():
                        other = resumed['optimizer']['state'][key][field]
                        self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)
                self.assertEqual(full['step'], 10)
                self.assertEqual(full['epoch'], 1)
                exported = torch.load(str(args.output / 'inference.pt'))
                clone = ToyModel().eval()
                clone.load_state_dict(exported['model'], strict=True)
                expected = ToyModel().eval()
                expected.load_state_dict(full['model'])
                x = torch.randn(1, 3, 56, 56)
                self.assertTrue(torch.equal(clone(x)[1][0], expected(x)[1][0]))
                args.seed = 2
                with self.assertRaisesRegex(ValueError, 'differs'):
                    train(args)
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                args.resume = None
                train(args)

    def test_04_filename_mask_pairing(self):
        with tempfile.TemporaryDirectory(dir=str(rt.ROOT / '.cache')) as tmp:
            root = Path(tmp)
            for defect in ['good', 'crack']:
                directory = root / 'bottle' / 'test' / defect
                directory.mkdir(parents=True)
                Image.new('RGB', (56, 56)).save(directory / '000.png')
            with self.assertRaisesRegex(ValueError, 'Missing mask'):
                MVTec(root, small_config(), 'test', 'bottle')
            directory = root / 'bottle' / 'ground_truth' / 'crack'
            directory.mkdir(parents=True)
            Image.new('L', (56, 56), 255).save(directory / '000_mask.png')
            dataset = MVTec(root, small_config(), 'test', 'bottle')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(sum(dataset[i][2] for i in range(2)), 1)

    def test_05_guards_and_sampler(self):
        self.assertEqual(batches(31, 4, 1, 0)[3:], batches(31, 4, 1, 0, 3))
        self.assertNotEqual(batches(31, 4, 1, 0), batches(31, 4, 1, 1))
        with self.assertRaisesRegex(ValueError, 'pretrained'):
            build(small_config())
        with tempfile.TemporaryDirectory(dir=str(rt.ROOT / '.cache')) as tmp:
            path = Path(tmp) / 'invalid.json'
            cfg = small_config('A1')
            cfg['noise_dropout'] = 0.2
            rt.write_json(path, cfg)
            with self.assertRaisesRegex(ValueError, 'mechanism'):
                rt.config(path)
            rt.write_json(path, dict(kind='synthetic-test', fixed_endpoint=True))
            with self.assertRaisesRegex(ValueError, 'real-data'):
                summarize(SimpleNamespace(evaluations=[path], output=Path(tmp) / 'summary'))

    def test_06_evaluation_summary_lifecycle_and_guards(self):
        with tempfile.TemporaryDirectory(dir=str(rt.ROOT / '.cache')) as tmp:
            root = Path(tmp)
            cfg = small_config()
            for cat in rt.CATEGORIES:
                for defect in ['good', 'crack']:
                    directory = root / 'data' / cat / 'test' / defect
                    directory.mkdir(parents=True)
                    pixels = np.random.RandomState(0 if defect == 'good' else 1).randint(0, 255, (56, 56, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(directory / '000.png')
                directory = root / 'data' / cat / 'ground_truth' / 'crack'
                directory.mkdir(parents=True)
                mask = np.zeros((56, 56), dtype=np.uint8)
                mask[14:42, 14:42] = 255
                Image.fromarray(mask).save(directory / '000_mask.png')
            saved = dict(model=ToyModel().state_dict(), config=cfg, step=cfg['steps'],
                         provenance=dict(seed=1, purpose='formal', source=rt.source_state(),
                                         encoder_sha256='synthetic', dataset_sha256='synthetic'))
            checkpoint = root / 'synthetic.pt'
            torch.save(saved, str(checkpoint))
            args = SimpleNamespace(checkpoint=checkpoint, data=root / 'data', output=root / 'eval',
                                   device='cpu', allow_incomplete=False)
            with patch('course.evaluate.build', side_effect=lambda *a, **kw: ToyModel()), contextlib.redirect_stdout(io.StringIO()):
                evaluate(args)
            evaluation = args.output / 'evaluation.json'
            summary_args = SimpleNamespace(evaluations=[evaluation], output=root / 'summary')
            summarize(summary_args)
            self.assertTrue((root / 'summary' / 'aggregate.csv').is_file())
            source = json.loads((root / 'summary' / 'sources.json').read_text(encoding='utf-8'))
            self.assertEqual(len(source['missing_planned_runs']), 8)
            with np.load(str(args.output / 'bottle.npz')) as bundle:
                self.assertEqual(bundle['maps'].shape, (2, 16, 16))
                self.assertEqual(len(bundle['paths']), 2)
            summary_args.evaluations = [evaluation, evaluation]
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize(summary_args)
            altered = copy.deepcopy(json.loads(evaluation.read_text(encoding='utf-8')))
            altered['protocol']['eval_size'] = 99
            alternate = args.output / 'altered.json'
            rt.write_json(alternate, altered)
            summary_args.evaluations = [evaluation, alternate]
            with self.assertRaisesRegex(ValueError, 'Mixed'):
                summarize(summary_args)
            altered['protocol']['eval_size'] = cfg['eval_size']
            altered['training']['purpose'] = 'probe'
            rt.write_json(alternate, altered)
            summary_args.evaluations = [alternate]
            with self.assertRaisesRegex(ValueError, 'Probe'):
                summarize(summary_args)
            saved['step'] = 1
            torch.save(saved, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'fixed endpoint'):
                evaluate(args)


    def test_07_edge_defect_validation(self):
        from run import main
        with tempfile.TemporaryDirectory(dir=str(rt.ROOT / '.cache')) as tmp:
            root = Path(tmp)
            for split, defect in [('train', 'good'), ('test', 'good'), ('test', 'crack')]:
                directory = root / 'data' / 'bottle' / split / defect
                directory.mkdir(parents=True)
                Image.new('RGB', (56, 56)).save(directory / '000.png')
            directory = root / 'data' / 'bottle' / 'ground_truth' / 'crack'
            directory.mkdir(parents=True)
            mask = Image.new('L', (56, 56))
            mask.paste(255, (0, 0, 3, 3))
            mask.save(directory / '000_mask.png')
            cfg = small_config()
            cfg['crop_size'] = 28
            rt.write_json(root / 'config.json', cfg)
            argv = ['run.py', 'validate-data', '--data', str(root / 'data'),
                    '--config', str(root / 'config.json'), '--output', str(root / 'validation.json')]
            with patch('course.data.CATEGORIES', ['bottle']), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((root / 'validation.json').read_text())
            self.assertEqual(result['defects_removed_by_crop'], ['bottle/test/crack/000.png'])
            self.assertEqual(len(result['test']), 2)
            Image.new('L', (56, 56)).save(directory / '000_mask.png')
            (root / 'validation.json').unlink()
            with patch('course.data.CATEGORIES', ['bottle']), patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(ValueError, 'Empty original'):
                    main()


if __name__ == '__main__':
    (rt.ROOT / '.cache').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
