import argparse

from ultralytics import RTDETR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CFS-DETR')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()

    model = RTDETR(args.weights)
    model.val(data=args.data,
              imgsz=args.imgsz,
              batch=args.batch,
              device=args.device)
