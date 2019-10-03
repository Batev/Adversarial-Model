import numpy as np

SEED = 33
IMG_HEIGHT = 28
IMG_WIDTH = 28
TEST_DATA_DIR = "data/digit-recognizer/"

def glue_images(images, x, y, margin=2):
    """
    Glue a number of images, in order to be able to
    plot them.
    :param images: All images to be glued.
    :param x: Number of images glued on the x-axis.
    :param y: Number of images glued on the y-axis.
    :param margin: Plot margin.
    :return: All images glued together.
    """

    img_width = images[0].shape[0]
    img_height = images[0].shape[1]

    width = x * img_width + (x - 1) * margin
    height = y * img_height + (y - 1) * margin
    glued_images = np.zeros((width, height, 3))

    for k in range(x):
        for j in range(y):
            img = images[k * y + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            glued_images[
                            (img_width + margin) * k: (img_width + margin) * k + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return glued_images

