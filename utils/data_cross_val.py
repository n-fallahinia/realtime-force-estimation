"""
functions for n fold cross validation of 
the training dataset
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

from __future__ import print_function

import glob 
import os  
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image  
import PIL 