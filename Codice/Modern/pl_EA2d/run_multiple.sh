#!/bin/bash

#Run multiple times the pseudolikelihood_EA2d.py
#Usage: run_multiple.sh <device_number> <number_repeats> <temperature>
#device_number: the number of the GPU to use
#number_repeats: the number of times to run the program
#temperature: the temperature of the data to use

# Check if exactly three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <device_number> <number_repeats> <temperature>"
    exit 1
fi

# Assign command line arguments to variables
DEVICE_NUMBER=$1
NUMBER_REPEATS=$2
TEMPERATURE=$3

# Check if the second argument is a positive integer
if ! [[ "$NUMBER_REPEATS" =~ ^[0-9]+$ ]]; then
    echo "Error: <number_repeats> must be a positive integer."
    exit 1
fi

# Activate the Conda environment 'old'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate old

# Set the visible device to the provided number
export CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER

# Repeat the launch of the Python program the specified number of times
for ((i=1; i<=NUMBER_REPEATS; i++))
do
    python3 pseudolikelihood_EA2d.py --temperature $TEMPERATURE
done
