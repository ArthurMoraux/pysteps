"""
pysteps.extrapolation.dgmr
====================================

Implementation of the deep learning based method DGMR described in :cite:`Ravuri2021`.

.. autosummary::
    :toctree: ../generated/

    extrapolate

"""

from dgmr import DGMR
import os
from torch import load
import urllib

def get_pretrained():
    """Download the weights of the DGMR model, or load them if already previously
    downloaded, and return the model with weights loaded.
    """
    
    MODELFOLDER = "../model_weights"
    os.makedirs(MODELFOLDER, exist_ok=True)
    MODELFILE = os.path.join(MODELFOLDER, "pytorch_model.bin")
    
    # If pytorch_model.bin file hasn't been download yet...
    if not os.path.isfile(MODELFILE):
        # Download from link found on https://huggingface.co/openclimatefix/dgmr/tree/main
        MODEL_URL = "https://huggingface.co/openclimatefix/dgmr/resolve/main/pytorch_model.bin?download=true"
        urllib.request.urlretrieve(MODEL_URL, MODELFILE)
        
    state_dict = load(MODELFILE)
    model = DGMR()
    model.load_state_dict(state_dict)
    
    return model

