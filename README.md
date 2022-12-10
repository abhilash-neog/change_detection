# Understanding Granularity in Change Detection

## How to run the object level change detection model (tiny_model_4_CD)
First cd into the Tiny_model_4_CD directory
``` 
$cd Tiny_model_4_CD
```
#### Downloading the LEVIR dataset
Downloading the dataset can be done 2 ways, either directly:
``` 
$wget https://www.dropbox.com/s/h9jl2ygznsaeg5d/LEVIR-CD-256.zip 
$unzip LEVIR-CD-256.zip
```
Or by using the executor script
```
$./executor.sh download
```
#### Training
Training can be run 2 ways, first by directly calling the python script with the following commands
```
$python3 training.py --datapath LEVIR-CD-256 --log-path log_path
```
or by using the executor script
```
$./executor.sh train
```
