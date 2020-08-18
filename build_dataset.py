"""
Script to split the Fingernail dataset into train/dev/test and
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

"""Split the Fingernail dataset into train/dev/test and .
The Fingernail dataset comes in the following format:
    subj_01/
        aligned_images/
            aligned_0001.mat
            ...
        raw_images/
            img_01_0001.jpg
            ...
        images/
            image_0001.jpg
            ...
        forces/
            force_01.txt
    ...

Original raw images have size (1024, 768). Aligned images have size (290, 290)
Using aligned images will reduce the size of the training dataset and the model will be more 
accurate.Test set is already created so it only splits the data to "train" and "eval" data sets. 
different methods of splitting can be used based on the parser argument. Here are each method:

1- Hyper-parameters tuning: in this case the entire subject data set will be used with 80%
    for training and 20% for evaluation. 
2- Optimality: To find out how many human subjects should be in traing ste in order to optimize 
   the estimation error, 3 subjects are selected randomly for evals set and a different
   configuration of subjects will be used for training eg. 5, 10, 12, 15.
3- Generalizability: To figure out how many human subjects needed in test dataset to make statistically 
   significant conclusions about estimation error.
"""

import argparse
import os
import sys

from model.utils.utils import Params
from model.utils.data_util import *

parser = argparse.ArgumentParser(description ='Build Fingernail dataset')
parser.add_argument('--mode', default='hyper', 
                    help="Hyper-parameters tuning mode")

parser.add_argument('--data_dir', default='./dataset',
                    help="Directory containing the dataset")

parser.add_argument('--output_dir', default='./data',
                    help="Where to write the new data")
                    
parser.add_argument('--model_dir', default='./experiments',
                    help="Experiment directory containing params.json")

parser.add_argument('--use_raw', default='false',
                    help="Experiment directory containing params.json")
                
parser.add_argument('-v', dest ='verbose', 
                    action ='store_true', help ='verbose mode')

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.output_dir, 'train')
    test_data_dir = os.path.join(args.output_dir, 'test')
    eval_data_dir = os.path.join(args.output_dir, 'eval')
    dataset_dir = args.data_dir

    filenames = {'train': train_data_dir,
                'eval': eval_data_dir,
                'test': test_data_dir,
                'data': dataset_dir}

    # Check that we are not overwriting some previous experiment
    if  os.path.exists(args.output_dir):
        print ("[Error]: output dir {} already exists !".format(args.output_dir)) 
        sys.exit()

    print('[INFO] Making the data directory ... ')
    os.mkdir(args.output_dir)
    os.mkdir(train_data_dir)
    os.mkdir(test_data_dir)
    os.mkdir(eval_data_dir)
            
    if args.mode == 'hyper':
        print('[INFO] Selected mode is: {}'.format(args.mode))
        subjIdx_list = list(range(1,18))
        load_train_withIdx(subjIdx_list, filenames)
        print('[INFO] Done building data')
    
    if args.mode == 'opt':
        # TODO
        pass

    if args.mode == 'gen':
        # TODO
        pass

