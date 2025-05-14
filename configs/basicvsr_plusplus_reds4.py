_base_ = ['../_base_/models/basicvsr_plusplus.py']  # adjust path to your base config

# Inherit most settings; only override input resolution flag
model = dict(
    type='BasicVSRPlusPlus',
    is_low_res_input=False,  # keep input/output at same resolution
)

# Dataset or test pipeline is unused here (we use custom script)
