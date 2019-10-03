import tensorflow as tf
import pandas as pd
import numpy as np
import random
import commons as cs

from commons import glue_images
from sklearn.model_selection import train_test_split
from keras import backend
from keras.models import load_model
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from matplotlib import pyplot as plt

# Predefined constants
MODEL_NAME = "mnist_model.h5"
PLT_HEIGHT = 12.0
PLT_WIDTH = 12.0
TEST_FILE = cs.TEST_DATA_DIR + "train.csv"
VALIDATION_DATA_SIZE = 4200

plt.rc('figure', figsize=(PLT_HEIGHT, PLT_WIDTH))

backend.set_learning_phase(False)
# Load the pre-trained model
keras_model = load_model(MODEL_NAME)

raw_data = pd.read_csv(TEST_FILE)

# Split data into training (90%) and validation (10%) sets
train, validate = train_test_split(raw_data,
                                   test_size=0.1,
                                   random_state=cs.SEED,
                                   stratify=raw_data['label'])

# Split into input and output variables
x_validation = validate.values[:, 1:].reshape(VALIDATION_DATA_SIZE, cs.IMG_WIDTH, cs.IMG_HEIGHT, 1)
y_validation = validate.values[:, 0]

# Configure TF
tf.set_random_seed(random.randint(0, VALIDATION_DATA_SIZE))
sess = backend.get_session()
x = tf.placeholder(tf.float32, shape=(None, cs.IMG_WIDTH, cs.IMG_HEIGHT, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

# Evaluate the accuracy of the model
x_validation = x_validation.astype('float32')
x_validation /= 255

result = np.argmax(keras_model.predict(x_validation), axis=1)
acc = np.mean(np.equal(result, y_validation))

# Initialize the Fast Gradient Sign Method (FGSM)
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}

# Generate adversarial examples
adv_x = fgsm.generate_np(x_validation, **fgsm_params)

adv_result = np.argmax(keras_model.predict(adv_x), axis=1)
adv_acc = np.mean(np.equal(adv_result, y_validation))

print("The normal validation accuracy is: {}".format(acc))
print("The adversarial validation accuracy is: {}".format(adv_acc))


if len(x_validation) == len(adv_x):
    count = 0

    while count <= 2:
        i = random.randint(0, len(x_validation))

        x_sample = x_validation[i].reshape(cs.IMG_WIDTH, cs.IMG_HEIGHT)
        adv_x_sample = adv_x[i].reshape(cs.IMG_WIDTH, cs.IMG_HEIGHT)

        adv_comparison = glue_images([x_sample, adv_x_sample], 1, 2)

        plt.imshow(adv_comparison)
        plt.show()

        normal_digit_img = x_sample.reshape(1, cs.IMG_WIDTH, cs.IMG_HEIGHT, 1)
        adv_digit_img = adv_x_sample.reshape(1, cs.IMG_WIDTH, cs.IMG_HEIGHT, 1)

        normal_digit_pred = np.argmax(keras_model.predict(normal_digit_img), axis=1)
        adv_digit_pred = np.argmax(keras_model.predict(adv_digit_img), axis=1)

        print('The prediction for the original number is: {}'.format(normal_digit_pred))
        print('The prediction for the adversarial example is: {}'.format(adv_digit_pred))

        count = count + 1
