"""
functions for visualizing the train, test, 
and the entire data set
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

from __future__ import print_function
from data_util import *

import glob 
import os  
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d 
from PIL import Image  


def force_show(sub_Idx):
    """
    show the force 3D force plot for each human subject

    Inputs:
    - subIdx: Index of the subject to be shown
    """
    force_path = './dataset/subj_' + f'{sub_Idx:02d}'+ '/forces/force_' + f'{sub_Idx:02d}' + '.txt'
    image_path = './dataset/subj_' + f'{sub_Idx:02d}'+ '/images/'
    force_num = len(glob.glob(image_path + '*.jpg'))
    force_list = load_force_txt(force_path,force_num)
    print('showing '+f'{force_num:03d}'+ ' raw forces for subject ' + f'{sub_Idx:02d}')

    fig = plt.figure(figsize = (10, 7)) 
    ax = plt.axes(projection ="3d") 

    for x, y, z in force_list:
        ax.scatter3D(x, y, z, color = "green")
    ax.set_xlabel('X-axis', fontweight ='bold')  
    ax.set_ylabel('Y-axis', fontweight ='bold')  
    ax.set_zlabel('Z-axis', fontweight ='bold')
    plt.title("3D force data") 
    plt.show()

def train_force_show():
    force_path = './dataset/train/forces/force.txt'
    image_path = './dataset/train/image/'
    force_num = len(glob.glob(image_path + '*.jpg'))
    force_list = load_force_txt(force_path,force_num)
    print('showing '+f'{force_num:03d}'+ ' raw forces for the train data')


    fig = plt.figure(figsize = (10, 7)) 
    ax = plt.axes(projection ="3d") 

    for x, y, z in force_list:
        ax.scatter3D(x, y, z, color = "green")
    ax.set_xlabel('X-axis', fontweight ='bold')  
    ax.set_ylabel('Y-axis', fontweight ='bold')  
    ax.set_zlabel('Z-axis', fontweight ='bold')
    plt.title("3D force data") 
    plt.show()

def test_force_show():
    force_path = './dataset/test/forces/force.txt'
    image_path = './dataset/test/image/'
    force_num = len(glob.glob(image_path + '*.jpg'))
    force_list = load_force_txt(force_path,force_num)
    print('showing '+f'{force_num:03d}'+ ' raw forces for the test data')


    fig = plt.figure(figsize = (10, 7)) 
    ax = plt.axes(projection ="3d") 

    for x, y, z in force_list:
        ax.scatter3D(x, y, z, color = "green")
    ax.set_xlabel('X-axis', fontweight ='bold')  
    ax.set_ylabel('Y-axis', fontweight ='bold')  
    ax.set_zlabel('Z-axis', fontweight ='bold')
    plt.title("3D force data") 
    plt.show()

def train_image_show(img_num=15):
    image_path = './dataset/train/image/'
    images_list = glob.glob(image_path + '*.jpg')
    mask = np.random.choice(range(len(images_list)), img_num, replace=False)
    images_to_show = [images_list[i] for i in mask]

    fig=plt.figure(figsize=(9, 6))
    columns = 5
    rows = 3
    for i in range(1, columns*rows +1):
        img = Image.open(images_to_show[i-1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off') 
        plt.title('Img_'+f'{mask[i-1]:03d}')   
    plt.show()


if __debug__:
    print('debug')
    sub_Idx = 5
    train_image_show()