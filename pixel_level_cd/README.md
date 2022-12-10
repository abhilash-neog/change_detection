
The pixel_level_cd model is based off of the github repository by Ragav Sachdeva (https://github.com/ragavsachdeva/The-Change-You-Want-to-See)

## How to run the object level change detection model (pixel_level_cd)
#### 1. Clone the repo

Move into the pixel_level_cd directory
``` 
$cd pixel_level_cd
```
#### 2. Download the LEVIR dataset

The dataset can be downloaded from here : https://justchenhao.github.io/LEVIR/

#### 3. Dataset directory

Create directory "data_dir" and move the dataset into the newly created directory

Create a copy of the labels dir, such that there are now 2 directories -> label_1 and label_2. This needs to be done for all train, val, test

#### 4. Training

Update the data paths in the config file ("./configs/change_detection_config.yml")

```
python executor.py --method segmentation --gpus 1 --config_file ./configs/change_detection_config.yml --max_epochs 100 --decoder_attention scse --wandb_id <wandb_id> --experiment_name <name_of_the_wandb_experiment>
```

-- wandb_id (weights and biases profile id) <br>
-- name_of_the_wandb_experiment (name of the run to be logged in weights and biases) <br>

Few more args that can be useful

--no_logging (disables logging to weights and biases) <br>
--quick_prototype (runs 1 epoch of train, val and test cycle with 2 batches) <br>
--resume_from_checkpoint <path> <br>
--load_weights_from <path> (initialises the model with these weights) <br>

Note: if logging is enabled, update the "save_dir" (where the logs are stored locally) in executor.py
  
#### 5. Testing
```
python executor.py --method segmentation --gpus 1 --config_file configs/change_detection_config.yml --decoder_attention scse --test_from_checkpoint <path>
```
-- path (provide the path to a checkpoint)

A sample checkpoint (from the experimentations) can be downloaded from here: 
