
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

Update the data paths in the config file ("./configs/change_detection_config.yml")

```
python executor.py --method segmentation --gpus 1 --config_file ./configs/change_detection_config.yml --max_epochs 100 --decoder_attention scse --wandb_id <wandb_id> --experiment_name <name_of_the_wandb_experiment>
```

-- wandb_id (weights and biases profile id)
-- name_of_the_wandb_experiment (name of the run to be logged in weights and biases)

Few more args that can be useful

--no_logging (disables logging to weights and biases)
--quick_prototype (runs 1 epoch of train, val and test cycle with 2 batches)
--resume_from_checkpoint <path>
--load_weights_from <path> (initialises the model with these weights)


#### 5. Testing
```
python executor.py --method segmentation --gpus 1 --config_file configs/change_detection_config.yml --decoder_attention scse --test_from_checkpoint <path>
```
-- path (provide the path to a checkpoint)
