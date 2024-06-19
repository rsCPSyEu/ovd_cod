# Open-vocabulary vs. Closed-set Object Detection


## Overview
This repository hosts the dataset links and evaluation code for **"Open-vocabulary vs. Closed-set: Best Practice for Few-shot Object Detection Considering Text Describability"**. 

## Datasets
We repurpose existing **ODinW (Object Detection in the Wild)** datasets. 
For more details of the datasets, please refer to the original papers, [GLIP](https://arxiv.org/abs/2112.03857) and [ELEVATER](https://arxiv.org/abs/2204.08790). 


### Download Original Datasets
To download original ODinW datasets, run the following command; 

<pre>
mkdir ./datasets
python download.py --dataset_path ./datasets/odinw
</pre>


### Donload Our Few-shot Training Annotations 

Please visit our [dropbox](https://www.dropbox.com/scl/fo/18rdkaxwvvc4xw584hx9c/AN5dHc-3k9etlbtX9eMpgYc?rlkey=ryfnxyfzch1fmc5ms501sadbu&st=hox2q01i&dl=0) to download the corresponding annotation file for each random seed.
                           
The folder structure is as follows:

<pre>

── Dataset
    ├── aerial_maritime_drone_large
    │   ├── train_shot_1_seed_0.json
    │   ├── train_shot_3_seed_0.json
    │   ├── train_shot_5_seed_0.json
    │   ├── train_shot_10_seed_0.json
    │   └── ...
    │
    └── ...
</pre>


Then, put the files under ```datasets/odinw/fewshot_annotation_v1``` such as;
<pre>
── datasets
    └── odinw
        └── fewshot_annotation_v1
            ├── aerial_maritime_drone_large
            │   ├── train_shot_1_seed_0.json
            │   ├── train_shot_3_seed_0.json
            │   ├── train_shot_5_seed_0.json
            │   ├── train_shot_10_seed_0.json
            │   └── ...
            ├── aquarium
            └── ...
</pre>


## Evaluation Code

<!-- The evaluation code will be made available soon. -->
We also provide our evaluation codes. 

### Installation

Our implementation is based on repositories of [Detectron2](https://github.com/facebookresearch/detectron2) and [DyHead](https://github.com/microsoft/DynamicHead).  

- Setup conda environment

<pre>
env_name=ovd_cod
conda create -n ${env_name} python=3.8 -y
conda activate ${env_name}

conda install pytorch=1.10 torchvision torchaudio -c pytorch -y 
conda install -c conda-forge cudatoolkit-dev -y
conda install -c anaconda nltk numpy=1.23.1 -y
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers==4.19
pip install scipy
pip install pycocotools
pip install opencv-python

# assert that CUDA_HOME is set as /[HOME]/anaconda3/envs/${env_name}
python setup.py build develop --user
</pre>


## License
Please refer to the license provided by the original datasets. You can check the information [here](https://public.roboflow.com/object-detection).