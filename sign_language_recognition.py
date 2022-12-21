import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization,MaxPool2D
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, losses

for dirname, _, filenames in os.walk('../Train_Sign'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# Import train data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "Train_Sign", labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(50, 50), shuffle=True, seed=123,
    validation_split=0.2, subset="training"
)
# Import validation data
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "Train_Sign", labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(50, 50), shuffle=True, seed=123,
    validation_split=0.2, subset="validation"
)
# Class names 
dict_labels = {0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h",8:"i",9:"j",10:"k",11:"l",12:"m",13:"n",14:"o",15:"p",16:"q",17:"r",18:"s",19:"t",20:"u",21:"unkowen",22:"v",23:"w",24:"x",25:"y",26:"z"}

# Visulaize data
fig, ax = plt.subplots()
ax.bar("data",40500 ,color= 'b', label='Data')
ax.bar("train",32400 ,color= 'r', label='Train')
ax.bar("val",8100 ,color='g', label='Val')
leg = ax.legend();

# Build cnn 
model = Sequential()
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(50,50 ,3) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(27, activation = 'softmax'))

model.compile(optimizer='Adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
call = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True)

# Training the Model
fit = model.fit(train_data,validation_data=val_data,epochs=30,callbacks=[call])

# Serialize weights to HDF5
model.save("./model_sign_language.h5")

# Evaluate the Model
model.evaluate(val_data)

# Plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = fit.history['accuracy']
val_acc = fit.history['val_accuracy']
loss = fit.history['loss']
val_loss = fit.history['val_loss']
epochs = range(1, len(loss) + 1)

# Accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()

# Loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Test 
image_path = "./istockphoto-1182201692-612x612.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)

p = model.predict(img)

prediction = np.argmax(p[0],axis=-1)

prob = np.max(p[0],axis=-1)

print(dict_labels[prediction])
print('Probability: ', prob)

plt.imshow(new_img)
