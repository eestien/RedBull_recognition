# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import config as cf


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(150, 150))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 150, 150, 3)
    # center pixel data
    img = img.astype('float32')
    # img = img - [75.68, 74.0, 76.6]
    # img = img - [123.68, 116.779, 103.939]
    return img


# load an image and predict the class
def run_example():
    # load the image
    img = load_image(cf.img_to_pred)
    # load model
    model = load_model('model/redbull_model_vgg.h5')
    # predict the class
    result = model.predict(img)
    print(result[0])


# entry point, run the example
run_example()