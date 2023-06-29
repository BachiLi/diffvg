conda update -n base -c defaults conda
conda env remove -n diffvg
conda env create -n diffvg -f environment.yml
conda activate diffvg

git submodule update --init --recursive
pip3 install -r requirements.txt
python3 setup.py install