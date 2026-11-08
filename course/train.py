import json
from pathlib import Path
import time

from . import runtime as rt
import torch
from torch.utils.data import DataLoader
from .data import MVTec, manifest
from .model import build, parameters, optimizer_scheduler, loss_value


def batches(length, batch_size, seed, epoch, start=0):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(length, generator=generator).tolist()
    return [order[i:i + batch_size] for i in range(start * batch_size, length - batch_size + 1, batch_size)]


def save_checkpoint(path, model, optimizer, scheduler, cfg, step, epoch, batch, provenance):
    rt.atomic_save(path, dict(format_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), config=cfg, step=step, epoch=epoch,
                             batch=batch, rng=rt.rng_state(), provenance=provenance))


def restore_checkpoint(path, model, optimizer, scheduler, cfg, provenance):
    saved = torch.load(str(path), map_location='cpu')
    if saved['config'] != cfg or saved['provenance'] != provenance:
        raise ValueError('Resume config/seed/code/dataset/encoder differs from checkpoint')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    rt.restore_rng(saved['rng'])
    return saved['step'], saved['epoch'], saved['batch']


def train(args):
    cfg = rt.config(args.config)
    if args.seed < 0:
        raise ValueError('seed must be nonnegative')
    out = Path(args.output).resolve()
    resume = Path(args.resume).resolve() if args.resume else None
    if out.exists() and any(out.iterdir()) and resume is None:
        raise ValueError('Output is nonempty; use --resume or a new directory')
    if resume and resume.parent != out:
        raise ValueError('Resume in the original output directory')
    out.mkdir(parents=True, exist_ok=True)
    rt.seed_all(args.seed)
    dataset = MVTec(args.data, cfg, 'train')
    if len(dataset) < cfg['batch_size']:
        raise ValueError('Dataset smaller than a full training batch')
    data_manifest = manifest(dataset)
    provenance = dict(seed=args.seed, purpose=args.purpose, source=rt.source_state(),
                      encoder_sha256=rt.sha256(args.encoder),
                      dataset_sha256=__import__('hashlib').sha256(
                          json.dumps(data_manifest, sort_keys=True).encode()).hexdigest())
    model = build(cfg, args.encoder).to(args.device)
    optimizer, scheduler = optimizer_scheduler(model, cfg)
    step, epoch, batch = 0, 0, 0
    if resume:
        step, epoch, batch = restore_checkpoint(resume, model, optimizer, scheduler, cfg, provenance)
    rt.write_json(out / 'config.json', cfg)
    rt.write_json(out / 'train-data.json', data_manifest)
    rt.write_json(out / 'provenance.json', provenance)
    rt.write_json(out / ('environment-from-step-%d.json' % step), rt.environment())
    # Log segments remain explicit after interruption; checkpoints are the source of truth.
    log = (out / ('train-from-step-%d.jsonl' % step)).open('a', encoding='utf-8')
    start_time = time.perf_counter()
    start_step = step
    if str(args.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    try:
        while step < cfg['steps']:
            sampler = batches(len(dataset), cfg['batch_size'], args.seed, epoch, batch)
            # Independent generator keeps iterator construction from consuming dropout RNG on resume.
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg['workers'],
                                generator=torch.Generator().manual_seed(args.seed + epoch))
            model.train()
            model.encoder.eval()
            for images, _, _, _ in loader:
                images = images.to(args.device)
                optimizer.zero_grad()
                en, de = model(images)
                loss = loss_value(en, de, cfg, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss at step %d' % step)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters(model), cfg['grad_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite gradients at step %d' % step)
                lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                step += 1
                batch += 1
                record = dict(step=step, loss=float(loss.detach()), lr=float(lr), grad_norm=float(norm),
                              elapsed_seconds=time.perf_counter() - start_time)
                log.write(json.dumps(record) + '\n')
                log.flush()
                if step % 50 == 0 or step == cfg['steps']:
                    print(json.dumps(record), flush=True)
                if step % cfg['save_every'] == 0 or step == cfg['steps']:
                    save_checkpoint(out / 'last.pt', model, optimizer, scheduler, cfg, step, epoch, batch, provenance)
                if step == cfg['steps']:
                    break
            if step < cfg['steps']:
                epoch, batch = epoch + 1, 0
        elapsed = time.perf_counter() - start_time
        rt.atomic_save(out / 'inference.pt', dict(format_version=1, model=model.state_dict(), config=cfg,
                                                 step=step, provenance=provenance))
        rt.write_json(out / 'completion.json', dict(step=step, fixed_endpoint=step == cfg['steps'],
                      session_steps=step - start_step, session_seconds=elapsed,
                      peak_memory_bytes=torch.cuda.max_memory_allocated(args.device)
                      if str(args.device).startswith('cuda') else None,
                      inference_sha256=rt.sha256(out / 'inference.pt')))
    finally:
        log.close()
