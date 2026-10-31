"""Training, fixed-protocol evaluation and aggregation. No downloads or random-weight training."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--data', required=True)
    train.add_argument('--encoder', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--seed', type=int, default=1)
    train.add_argument('--device', default='cuda:0')
    train.add_argument('--resume')
    train.add_argument('--purpose', choices=['formal', 'probe'], default='formal')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--data', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--device', default='cuda:0')
    evaluate.add_argument('--allow-incomplete', action='store_true')
    summary = sub.add_parser('summarize')
    summary.add_argument('evaluations', nargs='+', help='Paths to evaluation.json')
    summary.add_argument('--output', required=True)
    validate = sub.add_parser('validate-data')
    validate.add_argument('--data', required=True)
    validate.add_argument('--config', default='configs/F.json')
    validate.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        from course.train import train
        train(args)
    elif args.command == 'evaluate':
        from course.evaluate import evaluate
        evaluate(args)
    elif args.command == 'summarize':
        from course.summarize import summarize
        summarize(args)
    else:
        from course import runtime as rt
        from course.data import MVTec, manifest
        cfg = rt.config(args.config)
        output = Path(args.output)
        if output.exists():
            raise ValueError('Validation output already exists')
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {}
        cropped_defects = []
        for split in ['train', 'test']:
            dataset = MVTec(args.data, cfg, split)
            for i in range(len(dataset)):
                image, mask, label, path = dataset[i]
                if label and not mask.bool().any():
                    from PIL import Image
                    with Image.open(dataset.samples[i][1]) as original:
                        if original.convert('L').getbbox() is None:
                            raise ValueError('Empty original defect mask: ' + path)
                    # Upstream center crop can remove edge defects; retain the official split.
                    cropped_defects.append(path)
            result[split] = manifest(dataset)
        result['defects_removed_by_crop'] = cropped_defects
        rt.write_json(output, result)
        if cropped_defects:
            print('WARNING: %d defect masks empty after official crop; paths recorded, labels retained.' % len(cropped_defects))
        print('Validated all 15 categories, image decoding, masks and file hashes.')


if __name__ == '__main__':
    main()
