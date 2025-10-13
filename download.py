# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained SiT models
"""
from torchvision.datasets.utils import download_url  
import torch
import os


pretrained_models = {'SiT-XL-2-256x256.pt'}


def find_model(model_name):
    """
    Finds a pre-trained SiT model.
    """
 
    assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint
    
def find_model_resume(model_name):
    """
    Finds a pre-trained ema SiT model. 
    """

    assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    return checkpoint


