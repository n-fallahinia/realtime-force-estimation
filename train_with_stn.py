"""
Script to train the NailNet model using the Fingernail data
This script is for the misaligned data
Make sure to run the "build_dataset.py" to creat the data folder
Navid Fallahinia - 07/11/2020
BioRobotics Lab
"""

import argparse
import os
import random
from packaging import version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model.input_fn import *
from model.model_align_fn import *
from model.training import *
from model.utils.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./experiments',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./data_5',
                    help="Directory containing the dataset")

parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

parser.add_argument('--log_dir', default="./log",
                    help="log directory for the trained model")

parser.add_argument('--stn_dir', default="./test/stn_model",
                    help="log directory for the trained model")

parser.add_argument('--mode', default='train', 
                    help="train or test mode")

parser.add_argument('--v', default=True,
                    help ='verbose mode')

if __name__ == '__main__':

    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)
    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

    # ste the gpu (device:GPU:0) 
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
            print(e)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # check if the data is available
    assert os.path.exists(args.data_dir), "No data file found at {}".format(args.data_dir)

    # check if the log file is available
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

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

    tf.debugging.set_log_device_placement(args.v)
    train_dataset = input_fn(True, images_list_train, force_list_train, params= params)
    eval_dataset  = input_fn(False, images_list_eval, force_list_eval, params= params)
    print('[INFO] Data pipeline is built')

    # Define the model
    print('=================================================')
    print('[INFO] Creating the model...')
    stn_module = tf.keras.models.load_model(args.stn_dir)
    model_spec = model_fn(args.mode, params, stn_module) 
    if args.v:
        model_spec['model'].summary()

    # Train the model
    print('=================================================')
    train_model = Train_and_Evaluate(model_spec, train_dataset, eval_dataset, args.log_dir)
    train_model.train_and_eval(params)
    print('=================================================')