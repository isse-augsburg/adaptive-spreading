# adaptive-spreading

Code for the paper "Learning Controllers for Adaptive Spreading ofCarbon Fiber Tows".
Download data (preprocessed) and a pretrained process model here:
https://figshare.com/s/1a3e9b1ac16362b46cf9

## Preprocessing

Scripts to preprocess the data are located in the subdirectory ./preprocessing.  
The major preprocessing steps are:
+ Removing NANs
+ Applying Savitzky-Golay-Filter
+ Building the average of consecutive measurements
+ Shifting the tow to the middle of each measurement

## Tow Prediction / Process Model

### Feedforward Neural Networks

### Random Forests

## Process Control Model

### Reward/Fitness Function

The reward function consists of three parts:
+ Target height
+ Target width
+ Bar movement

`cost = -(k_h * abs(target_height - mean_current_height) + k_w * abs(target_width - current_width) + k_m * total_bar_movement)`

The three criteria can be scaled individually.

### Algorithms


+ Genetic Algorithm (start_neuroevolution.py)
    + Implements a rather simplistic GA with mutation. No crossover capabilities are provided.
