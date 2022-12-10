
The pixel_level_cd model is based off of the github repository by Ragav Sachdeva (https://github.com/ragavsachdeva/The-Change-You-Want-to-See)

## How to run the object level change detection model (pixel_level_cd)
#### 1. Clone the repo

Move into the pixel_level_cd directory
``` 
$cd pixel_level_cd
```
#### 2. Download the LEVIR dataset

Downloading the dataset can be done 2 ways, either directly:
``` 
$wget https://www.dropbox.com/s/h9jl2ygznsaeg5d/LEVIR-CD-256.zip 
$unzip LEVIR-CD-256.zip
```

#### 3. Dataset directory

Create directory "data_dir" and move the dataset into the newly created directory

Create a copy of the labels dir, such that there are now, 2 dir -> label_1 and label_2. This needs to be done for all train, val, test

#### 4. Training
```
python main.py --method centernet --gpus 2 --config_file configs/detection_resnet50_3x_coam_layers_affine.yml --max_epochs 200 --decoder_attention scse
```
The codebase is heavily tied in with Pytorch Lightning and Weights and Biases. You may find the following flags helpful:

--no_logging (disables logging to weights and biases)
--quick_prototype (runs 1 epoch of train, val and test cycle with 2 batches)
--resume_from_checkpoint <path>
--load_weights_from <path> (initialises the model with these weights)
--wandb_id <id> (for weights and biases)
--experiment_name <name> (for weights and biases)

#### 5. Testing
```
python main.py --method centernet --gpus 2 --config_file configs/detection_resnet50_3x_coam_layers_affine.yml --decoder_attention scse --test_from_checkpoint <path>
```

Demo/Inference:

```
python demo_single_pair.py --load_weights_from <path_to_checkpoint> --config_file configs/detection_resnet50_3x_coam_layers_affine.yml --decoder_attention scse
```
