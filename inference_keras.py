from keras.models import load_model
import keras
import tensorflow as tf
# load model
import numpy as np
import os
from PIL import Image
model = load_model('rochin_model_grape_fine_tuned/')

print(model.summary())
IMAGE_SHAPE = (224, 224)
for  sunflower_path in os.listdir("Resaga"):
    #sunflower_path = "0_tomatoes_data_2021_2_25_9_15_41_V11_R1_COL_682.BMP"
    img_height = img_width = 224
    #img = keras.preprocessing.image.load_img(
    #  sunflower_path, target_size=(img_height, img_width)
    img = Image.open("Resaga/"+sunflower_path).resize(IMAGE_SHAPE)
    img = np.array(img).astype(np.float32)/ 255.0

    result = model.predict(np.expand_dims(img, axis=0))
    """img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)"""
    #score = tf.nn.softmax(predictions[0])
    print(result)
    predicted_class = np.argmax(result[0])
    print(predicted_class)