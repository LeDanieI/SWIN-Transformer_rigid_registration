# Swin-transformed based rigid registration model

Swin transformer for rigid image registration.

## Requirements

<pre><code>matplotlib==3.7.0
monai==1.1.0 
numpy==1.23.5
python==3.10.9
pytorch==1.13.1
scipy==1.10.0
SimpleITK==2.2.1
skimage==0.19.3
timm==0.6.12

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install scikit-image numpy
conda install -c simpleitk simpleitk
conda install -c conda-forge matplotlib tqdm
conda install -c conda-forge timm
conda install -c conda-forge einops
pip install ml-collections</code></pre>
  
## Data
OASIS 1 dataset retrieved from https://learn2reg.grand-challenge.org/Datasets/ or https://oasis-brains.org/
## Source code 
This project uses source code from:
1. Swin-Transformer code retrieved from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
2. TransMorph https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git
