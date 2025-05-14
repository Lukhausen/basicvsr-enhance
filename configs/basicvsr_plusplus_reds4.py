_base_ = ['./_base_/models/basicvsr_plusplus.py']

model = dict(
    type='BasicVSRPlusPlus',     # will be popped at runtime
    is_low_res_input=False,      # full-res in → full-res out
)
