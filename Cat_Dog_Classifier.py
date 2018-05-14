from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as keras_backend
import numpy as np
import helper_functions as hf

image_width = 64
image_height = 64
image_path = R'data/input/test'

if keras_backend.image_data_format() == 'channels_first':
    input_shape = (3, image_height, image_width)
else:
    input_shape = (image_height, image_width, 3)

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
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

classifier.load_weights('weights/updated_weights.h5py')

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data_generator = ImageDataGenerator(rescale=1./255)
image_generator = hf.get_absolute_filenames(image_path)

list_of_outputs = []

for item in image_generator:
    image_to_classify = load_img(item, target_size=(image_height, image_width))
    image_to_classify = img_to_array(image_to_classify)
    image_to_classify = np.expand_dims(image_to_classify, axis=0)
    list_of_outputs.append(classifier.predict(image_to_classify))

outputs = len(list_of_outputs)
names = list(hf.get_filenames(image_path))
prediction_and_filename = []

for i in range(outputs):
    prediction = list_of_outputs[i][0][0]
    if prediction == 1:
        prediction = 'Dog'
    elif prediction == 0:
        prediction = 'Cat'
    prediction_filename = names[i]
    prediction_and_filename.append((prediction, prediction_filename))

for prediction in prediction_and_filename:
    print(prediction)
