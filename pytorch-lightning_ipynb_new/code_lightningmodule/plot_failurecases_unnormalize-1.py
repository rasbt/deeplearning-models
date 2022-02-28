# Append the folder that contains the
# helper_data.py, helper_plotting.py, and helper_evaluate.py
# files so we can import from them

import sys

sys.path.append("../pytorch_ipynb")

from helper_data import UnNormalize
from helper_plotting import show_examples

# We normalized each channel during training; here
# we are reverting the normalization so that we
# can plot them as images
unnormalizer = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

show_examples(
    model=lightning_model,
    data_loader=test_dataloader,
    unnormalizer=unnormalizer,
    class_dict=class_dict,
)
