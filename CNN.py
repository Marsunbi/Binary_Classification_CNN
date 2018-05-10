from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image

img_width, img_height = 64, 64
img_path = R'prediction'

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.load_weights('updated_weights.h5py')

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

'''
#classifier.fit_generator(training_set, steps_per_epoch=1800, epochs=50, validation_data=test_set, validation_steps=800)
#classifier.save_weights('updated_weights.h5py')
'''

datagenerator = test_datagen.flow_from_directory(img_path, target_size=(img_width, img_height), class_mode='binary', shuffle=None)
result = classifier.predict_generator(datagenerator)
print(result)
print(result.shape)


'''
#for file in os.listdir('prediction'):
test_image = image.load_img('prediction/cat_or_dog.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    print('Dog ')
    print(result.shape)
else:
    print('Cat ')
    print(result)
'''