"""
Object365 dataset path is not set appropriately in the annotation file.
This script is to fix the path in the annotation file.
"""

import torch
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import glob

import multiprocessing

def loadjson(file):
    with open(file) as f:
        data = json.load(f)
    return data

def savejson(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
    return

# ==== [multi process]

def process_chunk(chunk, root, split):
    # """各チャンクの要素を10倍にして返す"""
    # return [i * 10 for i in chunk]

    root_directory = '{}/{}*'.format(root, split) # train_part* or val
    for imginfo in tqdm(chunk, desc='Searching for image paths'):
        filename = imginfo['file_name'] # only img name

        # 検索するパターン（例えば、'.txt' ファイル）
        pattern = '**/{}'.format(filename)

        # 再帰的に検索し、一致する全ファイルのフルパスを取得
        abs_file_path = glob.glob(f'{root_directory}/{pattern}', recursive=True)
        assert len(abs_file_path) == 1, 'len(abs_file_path)={}'.format(len(abs_file_path))
        abs_file_path = abs_file_path[0]
        assert os.path.isfile(abs_file_path), 'abs_file_path={} is not a file'.format(abs_file_path)        
        
        rel_filepath_from_root = abs_file_path.replace(root, '')
        if rel_filepath_from_root.startswith('/'):
            rel_filepath_from_root = rel_filepath_from_root[1:] # remove the first '/'
        imginfo['file_name'] = rel_filepath_from_root
    return chunk


def parallel_process(imginfo_list, num_splits, root, split):
    # リストを等分割
    size = len(imginfo_list) // num_splits + 1
    chunks = [imginfo_list[i:i + size] for i in range(0, len(imginfo_list), size)]

    # プールを作成し、各チャンクをプロセスに割り当て
    with multiprocessing.Pool(processes=num_splits) as pool:
        # results = pool.map(process_chunk, chunks)
        results = pool.starmap(process_chunk, [(chunk, root, split) for chunk in chunks])
    # 結果を結合
    return [item for sublist in results for item in sublist]


def get_new_annotations(json_res, root, split):    
    imginfo_list = json_res['images'] # len = N 

    num_splits = multiprocessing.cpu_count() 
    imginfo_list = parallel_process(imginfo_list, num_splits, root, split)
    print('after parallel_process: len(imginfo_list)={}'.format(len(imginfo_list)))
    assert len(imginfo_list) == len(json_res['images'])
    json_res['images'] = imginfo_list
    return json_res


def main(root, ann_files, save_files, splits):
    
    for (ann_file, save_file, split) in zip(ann_files, save_files, splits):
        old_json = loadjson(ann_file)
        new_json_images = get_new_annotations(old_json, root, split)
        savejson(new_json_images, save_file)
    return


if __name__ == '__main__':
    # dir to ann files
    # root = '/mnt/hdd03/img_data/o365_opendatalab/908e8445-437e-4686-86b2-51309ba6a6b9/raw/Objects365_v1/2019-08-02'
    root = './datasets/o365'
    ann_files = [
        os.path.join(root, 'objects365_train.json'),
        # os.path.join(root, 'objects365_val.json'),
    ]
    save_files = [
        os.path.join(root, 'objects365_train_correctpath.json'),
        # os.path.join(root, '_test.json'),
    ]
    splits = [
        'train', 
        # 'val'
    ]
    main(root, ann_files, save_files, splits)
    