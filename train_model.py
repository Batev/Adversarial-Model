from __future__ import print_function
import pandas as pd
import keras
import commons as cs
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as kb
from keras.preprocessing.image import ImageDataGenerator

TRAIN_FILE = cs.TEST_DATA_DIR + "train.csv"
BATCH_SIZE = 512
NUM_CLASSES = 10
EPOCHS = 32

raw_data = pd.read_csv(TRAIN_FILE)

train, validate = train_test_split(raw_data, test_size=0.1, random_state=cs.SEED, stratify=raw_data['label'])

# Split into input (X) and output (Y) variables
x_train = train.values[:, 1:]
y_train = train.values[:, 0]

x_validate = validate.values[:, 1:]
y_validate = validate.values[:, 0]

# Input image dimensions
img_rows, img_cols = cs.IMG_WIDTH, cs.IMG_HEIGHT

if kb.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_validate = x_validate.reshape(x_validate.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_validate = x_validate.reshape(x_validate.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_validate = x_validate.astype('float32')
x_train /= 255
x_validate /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_validate = keras.utils.to_categorical(y_validate, NUM_CLASSES)

# Use the built-in data generation
data_generator = ImageDataGenerator(
    width_shift_range=0.075,
    height_shift_range=0.075,
    rotation_range=12,
    shear_range=0.075,
    zoom_range=0.05,
    fill_mode='constant',
    cval=0
)

data_generator.fit(x_train)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.0001)

# Train the model
model.fit_generator(data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    steps_per_epoch=x_train.shape[0]/32,
                    verbose=1,
                    validation_data=(x_validate, y_validate),
                    callbacks=[reduce_lr])

score = model.evaluate(x_validate, y_validate, verbose=0)

print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# Save the model
#model.save("mnist_model.h5")
