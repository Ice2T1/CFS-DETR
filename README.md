# CFS-DETR

Official implementation of **CFS-DETR: Cross-Scale Invariant Feature Learning for Fiber Ultrastructure Detection**.

CFS-DETR is developed for cross-scale agglomeration detection in scanning electron microscopy (SEM) images. It contains:

- **FSE-Net**, including the Fiber SEM Edge Enhancer (FSEE), Universal Edge Enhancer (UEE), and Edge Selection Module (ESM).
- **CIM**, which uses a gradient reversal layer and a scale-domain classifier to learn cross-scale invariant features.

Only the code required for CFS-DETR is included here. The underlying detector is provided by Ultralytics and is not duplicated in this repository.

## Requirements

The code is based on **Ultralytics 8.0.201**. Install a PyTorch build compatible with your CUDA driver, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains only the direct runtime dependencies: PyTorch, TorchVision, NumPy, OpenCV, and PyYAML.

## Installation

Clone this repository and the official Ultralytics repository:

```bash
git clone https://github.com/Ice2T1/CFS-DETR.git
git clone --branch v8.0.201 https://github.com/ultralytics/ultralytics.git ultralytics-8.0.201
```

Apply the CFS-DETR integration patch and install Ultralytics in editable mode:

```bash
cd ultralytics-8.0.201
git apply ../CFS-DETR/patches/ultralytics-8.0.201-cfs-detr.patch
pip install -e .
cd ../CFS-DETR
```

## Dataset format

The dataset follows the Ultralytics YOLO detection format:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Edit `data_example.yaml` to point to the dataset. Scale-domain labels are inferred from image filenames:

- filenames containing `nm` are assigned to the nm-scale domain (domain 0);
- filenames containing `#U03bcm` or `μm` are assigned to the micrometer-scale domain (domain 1).

The DHU-OD-696 dataset is associated with an ongoing research project and is not publicly released at this stage.

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

## Core files

- `cfs_detr/fse_net.py`: FSE-Net, FSEE, UEE, and ESM.
- `cfs_detr/cim.py`: CIM, gradient reversal, and domain classification.
- `cfs_detr/domain.py`: nm/μm scale-domain labels.
- `configs/cfs_detr.yaml`: CFS-DETR architecture.
- `patches/ultralytics-8.0.201-cfs-detr.patch`: connection to RT-DETR training, loss, and forward propagation.

## License

The implementation is built on Ultralytics 8.0.201 and follows its AGPL-3.0 license.
