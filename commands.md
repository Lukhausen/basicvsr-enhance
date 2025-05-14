cd ..
rm basicvsr-enhance -r
clear
git clone https://github.com/Lukhausen/basicvsr-enhance
cd basicvsr-enhance
bash install.sh
tree -L 4
python run.py --input_folder examples/input --output_path examples/output/enhanced.png