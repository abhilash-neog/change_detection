#!/bin/bash

if [ "$1" == "test" ]; then
    echo test
fi
if [ "$1" == "train" ]; then
    python3 training.py --datapath LEVIR-CD-256 --log-path model_logs
fi
