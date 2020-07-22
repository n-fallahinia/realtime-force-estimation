# Load functions for prepairing the training data before
# feeding into the training session. Data can be load directly
# from the entire train set or can be iteraitively be loaded
# from the coss-validation folds.

# Navid Fallahinia - 06/16/2020
# BioRobotics Lab
# ===============================================================


from __future__ import print_function
from data_util import *

import glob
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

def load_dataset(data_path, IMG_WIDTH=224, IMG_HEIGHT=224, num_threads = 5):
    """
    creating a tf dataset pipeline that will be used later for 
    data augmutation and training the model

    Inputs:
    - train_path: path to the training images
    """
    if os.path.isfile(data_path):
        print('%s does not exists '% data_path)
        return
    image_path = data_path + 'image/'
    force_path = data_path + 'forces/force.txt'

    images_list = glob.glob(image_path + '*.jpg')
    force_list = load_force_txt(force_path,len(images_list))

    dataset = tf.data.Dataset.from_tensor_slices((images_list,force_list))
    if len(images_list) == 0:
        print('No images at this directory %s'% image_path)
        return
    print('*****************************')
    print('Dataset is built by %d images'% len(images_list))
    

    dataset = dataset.map(preprocess_data, num_parallel_calls=num_threads)
    dataset = dataset.shuffle(buffer_size=len(images_list))

def preprocess_data(image, force, WIDTH = 224, HEIGHT = 224 ):
    """ preprocessing images """   

    image = load_image(image, WIDTH, HEIGHT)
    image = augment_image(image)

    return image, force

def load_image(image, width, height):

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32) 
    image = tf.image.resize(image, [width, height]) 

    return image

def augment_image(image, FLIP_FLAG = False, COLOR_FLAG = False):
    
    if FLIP_FLAG:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    if COLOR_FLAG:
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image




if __debug__:
    print('DEBUG')
    train_path = 'dataset/test/' # Will be an arg in the data_load() function as a parser
    load_dataset(train_path)
