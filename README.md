# CFS-DETR

Official implementation of **CFS-DETR: Cross-Scale Invariant Feature Learning
for Fiber Ultrastructure Detection**.

This release is based directly on the Ultralytics 8.0.201 source tree used for
the experiments. It includes the complete forward-propagation, scale-domain
label, gradient-reversal, domain-loss, FSE-Net, and RT-DETR training paths.

## Installation

Use Python 3.8 and install a PyTorch build compatible with the local CUDA
driver. Then run:

```bash
pip install -r requirements.txt
pip install -e .
```

The CUDA implementation bundled at
`ultralytics/nn/extra_modules/ops_dcnv3/dist/` was built for Linux and Python
3.8. Install it only if the local environment reports that DCNv3 is missing.

## Dataset

The dataset must use the Ultralytics YOLO detection layout:

```text
dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Edit `data_example.yaml` before training. Scale-domain labels are generated
from image filenames: names containing `nm` are assigned to domain 0; names
containing `μm` or `#U03bcm` are assigned to domain 1.

The DHU-OD-696 dataset is associated with an ongoing research project and is
not publicly released at this stage.

## Training

```bash
python train.py --data /path/to/data.yaml --device 0 --epochs 150 --batch 4
```

Multi-GPU example:

```bash
python train.py --data /path/to/data.yaml --device 0,1,2,3 --epochs 150 --batch 16
```

## Evaluation

```bash
python val.py --weights /path/to/best.pt --data /path/to/data.yaml --device 0
```

## Main implementation locations

- `ultralytics/cfg/models/rt-detr/cfs-detr.yaml`: model configuration.
- `ultralytics/nn/extra_modules/domain_classifier.py`: Cross-Scale Invariant Module (CIM) and gradient reversal.
- `ultralytics/nn/extra_modules/block.py`: Fiber SEM Edge Enhancement Network (FSE-Net), Fiber SEM Edge Enhancer (FSEE), and Universal Edge Enhancer (UEE).
- `ultralytics/nn/extra_modules/attention.py`: Edge Selection Module (ESM).
- `ultralytics/nn/tasks.py`: CFS-DETR forward path and joint detection/domain loss routing.
- `ultralytics/data/dataset.py`: scale-domain label generation and batching.
- `ultralytics/models/utils/loss.py`: domain-adversarial loss.

## License

This implementation is based on Ultralytics 8.0.201 and follows the AGPL-3.0
license.
