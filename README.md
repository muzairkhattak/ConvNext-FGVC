# Benchmarking ConvNeXt models on FGVC datasets

This repository adds data-loader supports for 3 fine-grained visual classification datasets for training ConvNeXt. You can train ConvNeXt using:
<ul>
  <li>
    CUB dataset
  </li>
    <li>
    FoodX-251 dataset
  </li>
    <li>
    CUB + Stanford Dogs dataset (a custom dataset containing categories from both CUB and Dogs datasets providing 320 classes)
    </li>
  </ul>
  
This code repo is forked from official [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt)
This repo was mainly designed to make a strong ConvNeXt teacher model trained on FGVC dataset and use it to distill knowledge into DeiT model. Please refer here to our DeiT Knowledge distillation repository for complete project details.

## Requirements and Installation
We have tested this code on Ubuntu 20.04 LTS with Python 3.8. 

Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Datasets

<b> CUB Dataset </b>
Download CUB dataset from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view). Extract the file. Our code expects the dataset folder to have the following structure:

```
CUB_dataset_root_folder/
    └─ images
    └─ image_class_labels.txt
    └─ train_test_split.txt
    └─ ....
```
<b> FoodX Dataset </b>
Download FoodX dataset from [here](https://github.com/karansikka1/iFood_2019). After extracting the files, the root folder should have following structure:

```
FoodX_dataset_root_folder/
    └─ annot
        ├─ class_list.txt
        ├─ train_info.csv
        ├─ val_info.csv
    └─ train_set
        ├─ train_039992.jpg
        ├─ ....
    └─ val_set
        ├─ val_005206.jpg
        ├─ ....
```

<b> Stanford Dogs Dataset </b>
Download the dataset from [here](http://vision.stanford.edu/aditya86/ImageNetDogs/). The root folder should have following structure:

```
dog_dataset_root_folder/
    └─ Images
        ├─ n02092339-Weimaraner
            ├─ n02092339_107.jpg
            ├─ ....
        ├─ n02101388-Brittany_spaniel
            ├─ ....
        ├─ ....
    └─ splits
        ├─ file_list.mat
        ├─ test_list.mat
        ├─ train_list.mat

```


## Training 


To train a ConvNeXt-base (384x384) model on CUB dataset, run the following command
```bash
python main.py --model convnext_base --data-set CUB --drop_path 0.8 --input_size 384 --batch_size 16 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 60 --weight_decay 1e-8  --layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 --output_dir /path/to/save/checkpoints --finetune /path/to/convnext/pretrained/imagenet/weights.pth --data-path /path/to/dataset
```

To train a ConvNeXt-base (384x384) model on CUB+Dog dataset, run the following command
```bash
python main.py --model convnext_base --data-set CUB_DOG --drop_path 0.8 --input_size 384 --batch_size 16 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 60 --weight_decay 1e-8  --layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 --output_dir /path/to/save/checkpoints --finetune /path/to/convnext/pretrained/imagenet/weights.pth --data-path /path/to/CUB/and/Dog/dataset/seperated/by/space
```

<b>Note</b>: For CUB + DOG dataset, please provide both paths in the --data-set parameter, seperated by a space.

For example: /l/users/u21010225/AssignmentNo1/CUB/CUB_200_2011/ /l/users/u21010225/AssignmentNo1/dog/

Similarly to train ConvNeXt on FoodX dataset, replace  ```CUB``` by ```FOOD``` in the ```--data-set``` argument and corresponding dataset path in above sample commands.

## Evaluation
To evaluate a ConvNeXt trained model on CUB + DOG dataset, run the following command

```bash
python main.py --model convnext_base --data-set CUB_DOG --eval true --resume /path/to/trained/model --input_size 384 --drop_path 0.2 --data_path /path/to/CUB/and/Dog/dataset/seperated/by/space
```

--- 
Acknowledgement:
This repo is based on official [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt)

