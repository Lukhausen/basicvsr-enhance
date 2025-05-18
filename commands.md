
git clone https://github.com/Lukhausen/basicvsr-enhance
cd basicvsr-enhance
bash install.sh

python demo/restoration_by_ordered_file_name.py configs/basicvsr_plusplus_reds4.py chkpts/basicvsr_plusplus_reds4.pth data/test results/test

