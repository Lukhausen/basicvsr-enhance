# configs/basicvsr_plusplus_reds4.py

_base_ = ['_base_/models/basicvsr_plusplus.py']

model = dict(
    type='BasicVSRPlusPlus',
    is_low_res_input=False,  # keep input/output at same resolution
)

# (we don’t need a dataset/test pipeline for inference here)
