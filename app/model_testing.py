from keras.models import load_model
from PIL import Image
import numpy as np
import yaml
# from keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import os
import cv2 as cv

config = yaml.load(open(f"./train_classification_config.yaml", "r"))

test_photos_path = ''
test_images = os.listdir(test_photos_path)

predict = []
model = load_model('your model path')

submission = pd.DataFrame()
for i in test_images:
    image = cv.imread(f'{test_photos_path}/{i}')
    image = Image.fromarray(image.astype('uint8'), 'RGB').resize((320, 320))
    
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)

    submission = submission.append({'image':i, 'label':np.argmax(model.predict(image))})

