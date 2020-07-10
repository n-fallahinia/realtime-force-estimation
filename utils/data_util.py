"""
Utility functions for loading the train and test data
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

from __future__ import print_function

import glob 
import os  
import numpy as np
from sklearn.model_selection import train_test_split


def lead_train_withIdx(subjIdx_list, load_force = True, raw_images = False, train_ratio = 0.8):
    """
    load the training dataset from images in subject folders and store them 
    in the train folder

    Inputs:
    - subjIdx_list: list of integers for subject indecies [1,17]
    - load_force: Flag to wether load the coressponding labels or not
    """
    dataset_path = './dataset/'
    train_path = dataset_path + 'train'
    test_path = dataset_path + 'test'
    valid_path =  dataset_path + 'validation'
    image_lists = []
    data_params = {} # might not be used
    force_lists = np.zeros((1,3))
    print('Datat Processing started ... ')

    for subIdx in subjIdx_list:
        subj_path = dataset_path+'subj_'+f'{subIdx:02d}'
        print('Proccessing images for subject_' +f'{subIdx:02d}'+ ' ...', end =" ")

        if(not os.path.isdir(subj_path)):
            print(' no file exists for subject_' +f'{subIdx:02d}'+ ' !!')
            continue

        if raw_images:
            image_list = sorted(glob.glob(subj_path + '/raw_images/*.jpg'))
        else:
            image_list = sorted(glob.glob(subj_path + '/images/*.jpg'))    
               
        image_lists += image_list
        print(' Done!')

        if load_force:
            force_path = glob.glob(subj_path + '/forces/*.txt')    
            print('\tProccessing force for subject_' +f'{subIdx:02d}'+ ' ...')
            force_list = load_force_txt(force_path[0],len(image_list))
            force_lists = np.vstack((force_lists, force_list))

    force_lists = force_lists[1:,:]
    data_to_write = train_test_split_data(force_lists,image_lists,train_ratio)
    print('Processing Done! ')
    print("Number of train data :{0:4d}, Number of test data :{1:4d}"
                . format(len(data_to_write[0]), len(data_to_write[1]))) 


def load_force_txt(force_path, force_num, force_dim = 3):
    """ read force from a txt file """

    force_list = np.zeros((force_num,force_dim))
    if(not os.path.isfile(force_path)):
        print('\tno force file exists to load!')
        pass

    with open(force_path) as filestream: 
        for count, line in enumerate(filestream):
            currentline = line.split(",")
            force_list[count,:] = np.array([currentline[0], currentline[1], currentline[2]], dtype=float)

    return force_list

def train_test_split_data(force_lists, image_lists, train_ratio ,validation = True ):
    """ spliting data into 3 different set """

    assert len(force_lists) == len(image_lists), "images and forces have different size"
    mask = list(range(len(force_lists)))
    mask_train, mask_test = train_test_split(mask, train_size= train_ratio, shuffle=True)

    force_lists_train = force_lists[mask_train]
    force_lists_test = force_lists[mask_test]

    image_lists_train = [image_lists[i] for i in mask_train]   
    image_lists_test = [image_lists[i] for i in mask_test] 

    return [force_lists_train, force_lists_test, image_lists_train, image_lists_test]

def write_data(data_to_write, train_path, test_path):
    ################################################################################
    # TODO: should finish this function and also write a cross validation function #
    ################################################################################
    pass

if __name__ == "__main__":
    subjIdx_list = [1,3,5]
    lead_train_withIdx(subjIdx_list)