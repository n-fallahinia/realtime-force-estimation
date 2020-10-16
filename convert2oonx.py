"""
Script to convert the NailNet model from tf and Keras to oonx 
which can be used for both visualization with netron and inference
with nvidia TensorRT (speeds up the inference by 60%)
Navid Fallahinia - 07/11/2020
BioRobotics Lab
"""

import argparse
import os
import sys
from subprocess import check_call
from packaging import version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

import onnx
import tf2onnx
# import keras2onnx

PYTHON = sys.executable
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='./log/20201012-095601',
                    help="log directory containing the saved model")

parser.add_argument('--mode', default='tf', 
                    help="convert from keras or tensorflow model")

parser.add_argument('--output_dir', default='model.onnx', 
                    help="output model name")

parser.add_argument('--v', default=True,
                    help ='visualization with Netron')

if __name__ == '__main__':

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

    # Load the model file
    args = parser.parse_args()

    # check for the required mode
    if args.mode == 'tf':
        model_dir = os.path.join(args.model_dir, 'best_full_model_path')
        assert os.path.exists(model_dir), "No tf model found at {}".format(model_dir) 
        output_dir = os.path.join(args.model_dir, args.output_dir)

        # Launch training with this config
        cmd = "{python} -m tf2onnx.convert --saved-model {model_dir} --output {output_dir}".format(python=PYTHON,
            model_dir=model_dir, output_dir=output_dir)
        check_call(cmd, shell=True)
        
    elif args.mode == 'keras':
        model_path = args.model_dir
        assert os.path.isfile(model_path), "No tf model found at {}".format(model_path)  

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # TODO
        # model = tf.keras.models.load_model(model_path)
        # onnx_model = keras2onnx.convert_keras(model, model.name)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    else:
        print('invalid mode')

    onnx_model = onnx.load(output_dir)
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')

    if args.v:
        # Launch netron with this config
        cmd = "netron {output_dir}".format(output_dir=output_dir)
        check_call(cmd, shell=True)