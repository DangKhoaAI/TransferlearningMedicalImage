from dataloader import split_data,split_data_v3, create_gens
from customtrainingclass import MyCallback
from showresult import plot_training ,plot_confusion_matrix
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import time

data_dir = '/kaggle/input/sarscov2-ctscan-dataset' #dataset1 (covid- 2.5 image , 2 classes)
data_dir_2= '/kaggle/input/lung-cancer-histopathological-images' #dataset 2 (lungcancer-15k image , 3 classes)
data_dir_3='/kaggle/input/brain-tumor-mri-dataset' # dataset 3 (brain tummor- 7k image , 4 classes)

try:
    # Get splitted data
    #train_df, valid_df, test_df = split_data(data_dir)
    #train_df, valid_df, test_df = split_data(data_dir_2)
    train_df, valid_df, test_df = split_data_v3(data_dir_3)
    # Get Generators
    batch_size = 32
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)

except:
    print('Invalid Input')
