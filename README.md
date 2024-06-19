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



## Evaluation Code
The evaluation code will be made available soon.


## License
Please refer to the license provided by the original datasets. You can check the information [here](https://public.roboflow.com/object-detection).