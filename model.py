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
#input layer, define input shape
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'Same', data_format = 'channels_last', activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'Same', data_format = 'channels_last', activation = 'relu'))

model.add(MaxPool2D(pool_size = 2))

#reduce dimensionallity prior to dense layer
model.add(Flatten())
#output layer to compare to labels!
model.add(Dense(10, activation = "softmax"))


plot_model(model, to_file='model.png')

#define the optimizer used, chose this one for SPEEEEED
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#loss function to determine how bad our network is at classification
loss  = 'categorical_crossentropy'

model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

#reduce learning rate by 1/2 if accuracy isnt improved after 3 epochs using keras.callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

#increase when trying to get a good result, currently 1 for testing
epochs = 1

batch_size = 100

#config data augmenter
datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1 ,zoom_range = 0.1)

#calculates stats necessary to do augmentations
datagen.fit(X_train)




history  = model.fit_generator(datagen.flow(X_train, Y_train,batch_size = batch_size), epochs = epochs, verbose = 1 , validation_data = (X_val, Y_val), validation_freq  = 5)


# predict results
results = model.predict(test)


# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
