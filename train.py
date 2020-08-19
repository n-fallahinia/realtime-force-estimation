"""
Script to train the NailNet model using the Fingernail data
Make sure to run the "build_dataset.py" to creat the data folder
Navid Fallahinia - 07/11/2020
BioRobotics Lab
"""

import argparse
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model.input_fn import *
from model.model_fn import *
from model.training import *
from model.utils.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./experiments',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./data',
                    help="Directory containing the dataset")

parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

parser.add_argument('--log_dir', default="./log",
                    help="log directory for the trained model")

parser.add_argument('--mode', default='train', 
                    help="train or test mode")

parser.add_argument('--v', default=False,
                    help ='verbose mode')

if __name__ == '__main__':

    print('******DEBUG******')
    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # check if the data is available
    assert os.path.exists(args.data_dir), "No data file found at {}".format(args.data_dir)

    train_data_dir = os.path.join(args.data_dir, 'train')
    eval_data_dir = os.path.join(args.data_dir, 'eval')

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
    eval_filenames = [os.path.join(eval_data_dir, f) for f in os.listdir(eval_data_dir)]

    # Get the train images list
    images_list_train = glob.glob(train_filenames[0] + '/*.jpg')
    images_list_eval = glob.glob(eval_filenames[0] + '/*.jpg')

    # Get the label forces 
    force_list_train = load_force_txt(train_filenames[1]+ '/force.txt',len(images_list_train))
    force_list_eval = load_force_txt(eval_filenames[1]+ '/force.txt',len(images_list_eval))

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(images_list_train)
    params.eval_size = len(images_list_eval)

    # Create the two iterators over the two datasets
    print('=================================================')
    print('[INFO] Dataset is built by {0} training images and {1} eval images '
            .format(len(images_list_train), len(images_list_eval)))

    tf.debugging.set_log_device_placement(False)
    train_dataset = input_fn(True, images_list_train, force_list_train, params= params)
    eval_dataset  = input_fn(False, images_list_eval, force_list_eval, params= params)
    print('[INFO] Data pipeline is built')

    # Define the model
    print('=================================================')
    print('[INFO] Creating the model...')
    model_spec = model_fn(args.mode, params) 
    if args.v:
        model_spec['model'].summary()

    # Train the model
    print('=================================================')
    print('[INFO] Training started ...')
    train_model = Train_and_Evaluate(model_spec, train_dataset, eval_dataset, args.log_dir)
    train_model.train_and_eval(params)
    print('=================================================')