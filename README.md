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


### Download Our Few-shot Datasets for Training 

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


<!-- Then, put the files under ```datasets/odinw/fewshot_annotation_v1``` such as;
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
</pre> -->


## Evaluation

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

# install detectron2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
python -m pip install -e .

conda install setuptools=59.5.0 -c conda-forge

</pre>


### Checkpoints
We provide finetuned checkpoints for each dataset. Please donwload them from our [dropbox](https://www.dropbox.com/scl/fo/18rdkaxwvvc4xw584hx9c/AN5dHc-3k9etlbtX9eMpgYc?rlkey=ryfnxyfzch1fmc5ms501sadbu&st=hox2q01i&dl=0).


### Run evaluation
To evaluate the finetuning results with test data, follow the instruction below.

- For the number of few-shot samples, select one from ["1_200_8", "3_200_4", "5_200_2", "10_200_1"] and set it to ```shot```.
- For task configuration, see ```configs/odinw_configs.txt``` and set the config path to ```task_config```.
- For random seeds for few-shot sampling, select one from [0,1,2,3,4] and set ```run_seedv1```.
- For finetuned checkpoints, set the correct path to ```weight```.

Then, run the following;
<pre>
shot=1_200_8 # 1shot
# shot=3_200_4 # 3shot
# shot=5_200_2 # 5shot
# shot=10_200_1 # 10shot
IFS='_' read -r s epoch c <<< "$shot"

run_seedv1=0 # select from [0,1,2,3,4]

# Specify a dataset you want to evaluate. See configs/odinw_configs.txt.
# Example: use 'aquarium' dataset
task_config=configs/odinw_35/Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml

weight=/path/to/finetuned/ckpt

output_dir=/path/to/output/dir

DETECTRON2_DATASETS=datasets \
python tools/finetune.py \
--config configs/fewshot/odinw/_base_fullft.yaml \
--num-gpus 4 \
--resume \
--dist-url tcp://127.0.0.1:29500 \
--eval_only
FEWSHOT.BASE_CONFIG ${task_config} \
FEWSHOT.SHOT_EPOCH_COPY ${s},${epoch},${c} \
FEWSHOT.FREEZE_METHOD full_ft \
MODEL.WEIGHTS ${weight} \
SOLVER.IMS_PER_BATCH 4 \
SEED 42 \
SOLVER.AUTO_TERMINATE_PATIENCE 8 \
TEST.EVAL_EPOCH 1 \
OUTPUT_DIR ${output_dir} \
FEWSHOT.RUN_SEEDv1 ${run_seedv1}
</pre>


## License
Please refer to the license provided by the original datasets. You can check the information provided by [roboflow](https://public.roboflow.com/object-detection).