import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_names", default="all", type=str) # "all" or names joined by comma
argparser.add_argument("--dataset_path", default="datasets/odinw", type=str)
args = argparser.parse_args()

# root = "https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35"
root = 'https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35/{}.zip?download=true'

# ref: https://public.roboflow.com/object-detection

all_datasets = [
    "AerialMaritimeDrone", 
    "AmericanSignLanguageLetters", 
    "Aquarium",
    "BCCD",
    "ChessPieces",
    "CottontailRabbits",
    "DroneControl",
    "EgoHands",
    "HardHatWorkers",
    "MaskWearing",
    "MountainDewCommercial",
    "NorthAmericaMushrooms",
    "OxfordPets",
    "PKLot",
    "Packages", 
    "PascalVOC",
    "Raccoon",
    "ShellfishOpenImages",
    "ThermalCheetah",
    "UnoCards",
    "VehiclesOpenImages",
    "WildfireSmoke",
    "boggleBoards",
    "brackishUnderwater", 
    "dice",
    "openPoetryVision",
    "pistols",
    "plantdoc",
    "pothole",
    "selfdrivingCar",
    "syntheticFruit"
    "thermalDogsAndPeople", 
    "vector",
    "websiteScreenshots"
]

print('all_datasets: {}'.format(len(all_datasets)))

datasets_to_download = []
if args.dataset_names == "all":
    datasets_to_download = all_datasets
else:
    datasets_to_download = args.dataset_names.split(",")

for dataset in datasets_to_download:
    if dataset in all_datasets:
        print("Downloading dataset: ", dataset)
        
        path = root.format(dataset)
        cmd = 'wget {} -O {}/{}.zip'.format(path, args.dataset_path, dataset)
        print('cmd: {}'.format(cmd))
        os.system(cmd)
        
        cmd = 'unzip {}/{}.zip -d {}'.format(args.dataset_path, dataset, args.dataset_path)
        print('cmd: {}'.format(cmd))
        os.system(cmd)
        
        cmd = 'rm {}/{}.zip'.format(args.dataset_path, dataset)
        print('cmd: {}'.format(cmd))
        os.system(cmd)
    else:
        print("Dataset not found: ", dataset)