import csv
import json
from pathlib import Path
import numpy as np

from . import runtime as rt
from .evaluate import METRICS


def summarize(args):
    runs = []
    seen = set()
    reference = None
    mechanism = {'variant', 'attention', 'noise_dropout', 'supervision', 'loss', 'workers', 'save_every'}
    for filename in args.evaluations:
        path = Path(filename)
        entry = json.loads(path.read_text(encoding='utf-8'))
        if entry['kind'] != 'real-data-evaluation' or not entry['fixed_endpoint']:
            raise ValueError('Only complete fixed-endpoint real-data evaluations enter main tables')
        if rt.sha256(path.parent / 'metrics.csv') != entry['metrics_sha256']:
            raise ValueError('Metrics CSV checksum mismatch')
        cfg, provenance = entry['config'], entry['training']
        if provenance['purpose'] != 'formal':
            raise ValueError('Probe runs cannot enter formal main tables')
        signature = dict(config={k: v for k, v in cfg.items() if k not in mechanism},
                         protocol=entry['protocol'], test=entry['test_sha256'],
                         train=provenance['dataset_sha256'], encoder=provenance['encoder_sha256'],
                         source=provenance['source']['fingerprint'])
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError('Mixed training budgets, encoders, data, code or evaluation protocols')
        identity = (cfg['variant'], provenance['seed'])
        if identity in seen:
            raise ValueError('Duplicate variant/seed: ' + str(identity))
        seen.add(identity)
        categories = [r['category'] for r in entry['results']]
        if categories != rt.CATEGORIES + ['mean']:
            raise ValueError('Missing or reordered category results')
        for row in entry['results']:
            if not all(np.isfinite(row[k]) and 0 <= row[k] <= 1 for k in METRICS):
                raise ValueError('Invalid metric value')
            if row['category'] == 'mean':
                for key in METRICS:
                    if not np.isclose(row[key], np.mean([r[key] for r in entry['results'][:-1]])):
                        raise ValueError('Category mean mismatch')
            runs.append(dict(variant=identity[0], seed=identity[1], **row))
    aggregate = []
    for variant in ['B', 'F', 'A1', 'A2', 'A3']:
        for category in rt.CATEGORIES + ['mean']:
            rows = [r for r in runs if r['variant'] == variant and r['category'] == category]
            if not rows:
                continue
            result = dict(variant=variant, category=category, n=len(rows))
            for key in METRICS:
                result[key + '_mean'] = float(np.mean([r[key] for r in rows]))
                result[key + '_std'] = float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else ''
            aggregate.append(result)
    paired = []
    means = {(r['variant'], r['seed']): r for r in runs if r['category'] == 'mean'}
    for seed in sorted({s for v, s in means if v == 'B'} & {s for v, s in means if v == 'F'}):
        paired.append(dict(seed=seed, **{k + '_F_minus_B': means['F', seed][k] - means['B', seed][k] for k in METRICS}))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('runs.csv', runs), ('aggregate.csv', aggregate), ('paired.csv', paired)]:
        with (out / name).open('w', newline='', encoding='utf-8') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    rt.write_json(out / 'sources.json', dict(evaluations=[dict(path=str(Path(p).resolve()), sha256=rt.sha256(p))
                                                       for p in args.evaluations], comparison=reference,
                  missing_planned_runs=[dict(variant=v, seed=s) for v, seeds in
                      [('B', [1, 2, 3]), ('F', [1, 2, 3]), ('A1', [1]), ('A2', [1]), ('A3', [1])]
                      for s in seeds if (v, s) not in seen]))
