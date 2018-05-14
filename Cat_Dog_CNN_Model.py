from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras_backend

image_width = 64
image_height = 64
image_path = R'data/input'
shear_range = 0.2
zoom_range = 0.2
target_size = (64, 64)
batch_size = 32
steps_per_epoch = 1800
epochs = 50
number_validation_samples = 800

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

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=shear_range, zoom_range=zoom_range, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('data/training_set', target_size=target_size, batch_size=batch_size, class_mode='binary')
test_set = test_datagen.flow_from_directory('data/test_set', target_size=target_size, batch_size=batch_size, class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=(number_validation_samples // batch_size))

classifier.save_weights('FILENAME')
