# Append the folder that contains the
# helper_data.py, helper_plotting.py, and helper_evaluate.py
# files so we can import from them

import sys

sys.path.append("../pytorch_ipynb")

from helper_plotting import show_examples


show_examples(
    model=lightning_model, data_loader=test_dataloader, class_dict=class_dict
)
