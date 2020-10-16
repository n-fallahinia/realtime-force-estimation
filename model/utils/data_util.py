"""
Utility functions for loading the train and test data
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

import glob 
import os  
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image  
import PIL 

def load_train_withRnd(subNum ,load_force = True, raw_images = False, train_ratio = 0.8):
    """
    load the training dataset from images in subject folders and store them 
    in the train folder with random number of subIdx

    Inputs:
    - subNum: number of subjects to be sampled from
    - load_force: Flag to wether load the coressponding labels or not
    - raw_images: whther to load raw images or aligned images 
    - train_ratio: ratio of train to test data 
    """
    subjIdx_list = np.random.choice(range(17), subNum, replace=False)
    print('Subjects that are selected: %s' % subjIdx_list)
    load_train_withIdx(subjIdx_list)
    

def load_train_withIdx(subjIdx_list, filenames, load_force = True, raw_images = False, test_ratio = 0.2):
    """
    load the training dataset from images in subject folders and store them 
    in the train folder

    Inputs:
    - subjIdx_list: list of integers for subject indecies [1,17]
    - load_force: Flag to wether load the coressponding labels or not
    """
    dataset_path = filenames['data']
    train_path = filenames['train']
    test_path = filenames['test']
    eval_path =  filenames['eval']
    image_lists = []
    data_params = {} # might not be used
    force_lists = np.zeros((1,3))
    print('[INFO] Data Processing started ... ')

    for subIdx in subjIdx_list:
        subj_path = dataset_path+'/subj_'+f'{subIdx:02d}'
        print('\tProccessing images for subject_' +f'{subIdx:02d}'+ ' ...', end =" ")

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
    data_to_write_train = train_test_split_data(force_lists, image_lists, test_ratio)
    # force_lists_eval, force_lists_test, image_lists_eval, image_lists_test
    data_to_write_test = train_test_split_data(data_to_write_train[1], data_to_write_train[3], 0.5)
    print('[INFO] Processing Done! ')
    print("[INFO] Number of train data :{0:4d}, Number of eval data :{1:4d}, Number of test data :{2:4d} "
                . format(len(data_to_write_train[0]), len(data_to_write_test[0]),  len(data_to_write_test[1]))) 
    # data write part 
    write_data(data_to_write_train, data_to_write_test, train_path, eval_path ,test_path)

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

def train_test_split_data(force_lists, image_lists, test_ratio ,validation = True ):
    """ spliting data into 3 different set """

    assert len(force_lists) == len(image_lists), "images and forces have different size"
    mask = list(range(len(force_lists)))
    mask_train, mask_test = train_test_split(mask, test_size= test_ratio, shuffle=True)

    force_lists_train = force_lists[mask_train]
    force_lists_test = force_lists[mask_test]

    image_lists_train = [image_lists[i] for i in mask_train]   
    image_lists_test = [image_lists[i] for i in mask_test] 

    return [force_lists_train, force_lists_test, image_lists_train, image_lists_test]

def write_data(data_to_write_train, data_to_write_test, train_path, eval_path ,test_path):
    """ write datat to a txt file """
    force_lists_train = data_to_write_train[0]
    image_lists_train = data_to_write_train[2]

    force_lists_eval = data_to_write_test[0]
    image_lists_eval = data_to_write_test[2]

    force_lists_test = data_to_write_test[1]
    image_lists_test = data_to_write_test[3]

    filelist = list([train_path, eval_path, test_path])

    for file in filelist:
        force_path = os.path.join(file, 'forces')
        image_path = os.path.join(file, 'image')
        os.mkdir(force_path)
        os.mkdir(image_path)

    # force data
    with open(train_path+'/forces/force.txt', 'w') as filestream: 
        for force in force_lists_train:
            filestream.write("%s,%s,%s\n"%(force[0], force[1], force[2]))
    print('\tTrain forces saved! ')

    with open(eval_path+'/forces/force.txt', 'w') as filestream: 
        for force in force_lists_eval:
            filestream.write("%s,%s,%s\n"%(force[0], force[1], force[2]))
    print('\tEval forces saved! ')

    with open(test_path+'/forces/force.txt', 'w') as filestream: 
        for force in force_lists_test:
            filestream.write("%s,%s,%s\n"%(force[0], force[1], force[2]))
    print('\tTest forces saved! ')

    # image data
    for Idx, train_image in enumerate(image_lists_train):
        img = Image.open(train_image)
        img.save(train_path+'/image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);       
    print('\tTrain images saved! ')

    for Idx, eval_image in enumerate(image_lists_eval):
        img = Image.open(eval_image)
        img.save(eval_path+'/image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);  
    print('\tEval images saved! ')

    for Idx, test_image in enumerate(image_lists_test):
        img = Image.open(test_image)
        img.save(test_path+'/image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);  
    print('\tTest images saved! ')