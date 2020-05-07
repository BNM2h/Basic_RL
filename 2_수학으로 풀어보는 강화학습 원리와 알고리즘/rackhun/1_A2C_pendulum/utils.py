import numpy as np
import torch

def unpack_batch(batch):
    unpack = []
    for idx in range(len(batch)):
        unpack.append(batch[idx])

    unpack = torch.cat(unpack ,axis=0)

    return unpack
