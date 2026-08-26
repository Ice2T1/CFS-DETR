"""Scale-domain labels for nm- and micrometer-scale SEM images."""

from pathlib import Path

import torch


def generate_domain_labels(image_paths):
    labels = []
    for path in image_paths:
        filename = Path(str(path)).name
        if 'nm' in filename:
            label = 0
        elif '#U03bcm' in filename or 'μm' in filename:
            label = 1
        else:
            label = 0
        labels.append(label)
    return torch.tensor(labels, dtype=torch.long)
