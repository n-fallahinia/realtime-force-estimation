"""
Script to train the NailNet model using the Fingernail data
This script is for the misaligned data
Make sure to run the "build_dataset.py" to creat the data folder
Navid Fallahinia - 07/11/2020
BioRobotics Lab
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model.input_fn import *
from model.model_align_fn import *
from model.training import *
from model.utils.utils import Params

import tensorflow.compat.v2 as tf
import random
import logging
from packaging import version

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './experiments', 'Path to experiment directory containing params.json.')
flags.DEFINE_string('data_dir', './data_5', 'Path to directory containing the datase.')
flags.DEFINE_string('restore_from', None, 'directory or file containing weights to reload before training.')
flags.DEFINE_string('loging_dir', './log', 'log directory for the trained model.')
flags.DEFINE_string('stn_dir', './test/stn_model', 'directory for the stn modoul.')
flags.DEFINE_string('mode', 'train', 'train or test mode.')
flags.DEFINE_boolean('verbose', False, 'verbose mode.')

def main(unused_argv):

    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)
    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
    tf.get_logger().setLevel(logging.ERROR)
    # strategy = tf.compat.v2.distribute.MirroredStrategy()

    # ste the gpu (device:GPU:0) 
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
            print(e)

    # Load the parameters from json file
    json_path = os.path.join(FLAGS.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # check if the data is available
    assert os.path.exists(FLAGS.data_dir), "No data file found at {}".format(FLAGS.data_dir)

    # check if the log file is available
    if not os.path.exists(FLAGS.loging_dir):
        os.mkdir(FLAGS.loging_dir)

    train_data_dir = os.path.join(FLAGS.data_dir, 'train')
    eval_data_dir = os.path.join(FLAGS.data_dir, 'eval')

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
    stn_module = tf.keras.models.load_model(FLAGS.stn_dir)
    model_spec = model_fn(FLAGS.mode, params, stn_module) 
    if FLAGS.verbose:
        model_spec['model'].summary()

    # Train the model
    print('=================================================')
    train_model = Train_and_Evaluate(model_spec, train_dataset, eval_dataset, FLAGS.loging_dir)
    train_model.train_and_eval(params)
    print('=================================================')

if __name__ == '__main__':
    tf.compat.v1.app.run()