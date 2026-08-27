import argparse

from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description='Train CFS-DETR')
    parser.add_argument('--data', required=True, help='Dataset YAML path')
    parser.add_argument('--model', default='ultralytics/cfg/models/rt-detr/cfs-detr.yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', default='cfs_detr')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.model)
    model.train(data=args.data,
                cache=False,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                project='runs/train',
                name=args.name)
