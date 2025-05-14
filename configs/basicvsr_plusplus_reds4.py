_base_ = ['./_base_/models/basicvsr_plusplus.py']

# override only what we need (keep full‐res input/output)
model = dict(
    type='BasicVSRPlusPlus',
    is_low_res_input=False,
) 