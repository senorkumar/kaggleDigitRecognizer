from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#think of mapping Y = f(X), so X is the images (28x28 pixels) and Y is the labels

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

#create testing and validation set
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

#create the model!
model  = Sequential()
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'Same', data_format = 'channels_last', activation = 'relu'))


plot_model(model, to_file='model.png', show_shapes = True)
