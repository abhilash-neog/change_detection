# Understanding Granularity in Change Detection
The tiny_model_4_cd model is based off of the github repository by AndreaCodegoni (https://github.com/AndreaCodegoni/Tiny_model_4_CD)
## How to run the object level change detection model (tiny_model_4_CD)
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
#### Testing
Testing can be done manually or via the executor script. Two different tests are available, one lets you test an image individually and the other runs the test on all images in the test set.

Testing a single image can be done by:
```
$python3 test_single_img.py test_1_4.png model_13.pth
```
These image and model paths can be changed to reflect which image/model you with to test with. For the image path, only the image filename needs to be specified, and the program will search the LEVIR dataset for that image name.
Or by using the executor script:
```
$./executor.sh test_single
```
This results in a test.png being written to the folder with the test results

Testing all images in test set with direct command
```
$python3 test_ondata.py --datapath LEVIR-CD-256 --modelpath model_13.pth
```
Or by using the executor script
```
$./executor.sh test
```
