#Importing các thư viện
import joblib
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# Tiền xử lý dữ liệu 
# Training cho xử lý ảnh 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
                            'Train_Flower', target_size=(64, 64), batch_size=32, class_mode='categorical')
# Test cho xử lý ảnh 
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')
# Xây dựng model
cnn = tf.keras.models.Sequential()
# Xây dựng lớp Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 5, activation = 'softmax'))
cnn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Cải thiện mô hình
es=tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=3,
                                     verbose=1,  restore_best_weights=True)
rlronp=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=1,
                                             verbose=1)
callbacks=[es, rlronp]
cnn.fit(x = training_set, validation_data = test_set, epochs = 100, callbacks = callbacks)
# Lưu model lại 
joblib.dump(cnn, "model_flower.pkl")