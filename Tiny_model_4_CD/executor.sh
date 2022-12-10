#!/bin/bash

if [ "$1" == "test_single" ]; then
	python3 test_single_img.py test_1_4.png model_13.pth
fi
if [ "$1" == "test" ]; then
	python3 test_ondata.py --datapath LEVIR-CD-256 --modelpath model_13.pth	
fi
if [ "$1" == "train" ]; then
	python3 training.py --datapath LEVIR-CD-256 --log-path model_logs
fi
if [ "$1" == "download" ]; then
	wget https://www.dropbox.com/s/h9jl2ygznsaeg5d/LEVIR-CD-256.zip
	unzip LEVIR-CD-256.zip
fi
