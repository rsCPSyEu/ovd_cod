# Open-vocabulary vs. Closed-set Object Detection


## Overview
This repository hosts the dataset links and evaluation code for **"Open-vocabulary vs. Closed-set: Best Practice for Few-shot Object Detection Considering Text Describability"**. 


## Datasets
We repurpose existing **ODinW (Object Detection in the Wild)** datasets, using the images without any modifications. To download the images and original annotations, please follow the instructions provided to access each original resource.
For more details of the datasets, please refer to the original papers, [GLIP](https://arxiv.org/abs/2112.03857) and [ELEVATER](https://arxiv.org/abs/2204.08790). 


### Download Original Dataset Resoueces
To download original datasets, i.e., **ODinW (Object Detection in the Wild)**, please follow the instrution provided by official repositry of [GLIP's official repositry](https://github.com/microsoft/GLIP).


### Donload Annotations for Our Few-shot Samples 

Please visit our [dropbox](https://www.dropbox.com/scl/fo/18rdkaxwvvc4xw584hx9c/AN5dHc-3k9etlbtX9eMpgYc?rlkey=ryfnxyfzch1fmc5ms501sadbu&st=hox2q01i&dl=0) to download the corresponding annotation file for each random seed.
                           
The folder structure is as follows:

<pre>

── Dataset
    ├── aerial_maritime_drone_large
    │   ├── train_shot_1_seed_0.png
    │   ├── train_shot_3_seed_0.png
    │   ├── train_shot_5_seed_0.png
    │   ├── train_shot_10_seed_0.png
    │   └── ...
    │
    └── ...
</pre>

## Evaluation Code
The evaluation code will be made available soon.


## License
Please refer to the license provided by the original datasets. You can check the information [here](https://public.roboflow.com/object-detection).