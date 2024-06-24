#!/bin/bash


CONFIG_FILE='./configs/ein_seld/seld.yaml'

# ---------------------------------------  DCASE2022 Task 3  ------------------------------------------
# dev set
for DATASET in {'STARSS22','official',}
do
    python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_data' --dataset_type='dev' 
    python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_pit_label' --dataset_type='dev' 
    python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_indexes' --dataset_type='dev'
done

### Remember to change the corresponding hdf5 directory if you change the default hdf5 directory in the config file
mkdir -p ./_hdf5/dcase2022task3/label/frame/STARSS22/
find ./dataset/STARSS22/ -name '*.csv' -type f -print -exec cp {} ./_hdf5/dcase2022task3/label/frame/STARSS22/ \;

CUDA_VISIBLE_DEVICES=1 python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_scalar' 

# eval set
DATASET='STARSS22' 
python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_data' --dataset_type='eval' 
python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_indexes' --dataset_type='eval'

# # ---------------------------------------  DCASE2023 Task 3  ------------------------------------------
# for DATASET in {'STARSS23','official'}
# do
#     python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_data' --dataset_type='dev' 
#     python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_pit_label' --dataset_type='dev' 
#     python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_indexes' --dataset_type='dev'
# done

# ### Remember to change the corresponding hdf5 directory if you change the default hdf5 directory in the config file
# mkdir -p ./_hdf5/dcase2023task3/label/frame/STARSS23/
# find ./dataset/STARSS23/ -name '*.csv' -type f -print -exec cp {} ./_hdf5/dcase2023task3/label/frame/STARSS23/ \;

# CUDA_VISIBLE_DEVICES=1 python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_scalar' 

# # eval set
# DATASET='STARSS23' 
# python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_data' --dataset_type='eval' 
# python code/main.py -c $CONFIG_FILE --dataset=$DATASET preprocess --preproc_mode='extract_indexes' --dataset_type='eval'