import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sklearn

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#think of mapping Y = f(X), so X is the images and Y is the labels

#mapping input number to correct label
Y_train = train["label"]

#mapping input number to correct set of pixels
X_train = train.drop(labels  = ["label"], axis = 1)

X_train = X_train/255.0
test = test/255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#one hot encode Y train labels
Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
