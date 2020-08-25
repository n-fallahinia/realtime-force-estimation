"""
Script to evaluate the model using test data
Make sure to run the "build_dataset.py" to creat the data folder
Navid Fallahinia - 07/11/2020
BioRobotics Lab
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./experiments',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./data_single',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    
    args = parser.parse_args()

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

    # check if the data is available
    assert os.path.exists(args.data_dir), "No data file found at {}".format(args.data_dir)

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

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
    train_dataset = input_fn(False, images_list_test, force_list_test, params= params)