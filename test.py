"""
Script to evaluate the model using test data
Make sure to run the "build_dataset.py" to creat the data folder
Navid Fallahinia - 07/11/2020
BioRobotics Lab
"""

import argparse
import os
from packaging import version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model.input_fn import *
from model.evaluation import *
from model.utils.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./log/20200825-143805/',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./data_single',
                    help="Directory containing the dataset")

parser.add_argument('--param_dir', default='./experiments',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    
    # Set the random seed for the whole graph
    tf.random.set_seed(230)

    args = parser.parse_args()

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

    # check if the data is available
    assert os.path.exists(args.data_dir), "No data file found at {}".format(args.data_dir)

    # Load the parameters from json file
    json_path = os.path.join(args.param_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # check if the model directory is available
    assert os.path.exists(args.model_dir), "No model file found at {}".format(args.model_dir)
    model_path = os.path.join(args.model_dir, 'best_full_model_path')

    test_data_dir = os.path.join(args.data_dir, 'test')
    # Get the filenames from the train and dev sets
    test_filenames = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)]
    # Get the train images list
    images_list_test = glob.glob(test_filenames[0] + '/*.jpg')
    # Get the label forces 
    force_list_test = load_force_txt(test_filenames[1]+ '/force.txt',len(images_list_test))
    # Specify the sizes of the dataset we train on and evaluate on
    params.test_size = len(images_list_test)

    # Create the two iterators over the two datasets
    print('=================================================')
    print('[INFO] test data is built by {0} images'.format(len(images_list_test)))
    test_dataset = input_fn(False, images_list_test, force_list_test, params= params)

    # Open the saved  model from log file the model
    print('=================================================')
    loaded_model = tf.saved_model.load(model_path)
    print('[INFO] Model loaded...')

    # Test the model
    print('=================================================')
    test_model = Evaluate(loaded_model, test_dataset)
    test_model.test(params)
    print('=================================================')