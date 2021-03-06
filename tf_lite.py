from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model_file('model/redbull_model_vgg.h5')
tfmodel = converter.convert()
open("model_lite/model.tflite","wb").write(tfmodel)

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model 


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

def run_example():
    # load the image
    img = load_image('data/pred_imgs/not_pred.jpg')
    # load model
    model = load_model('model_lite/model.tflite')
    # predict the class
    result = model.predict(img)
    print(result[0])


# entry point, run the example
run_example()