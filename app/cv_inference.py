from keras.models import load_model
from PIL import Image
import numpy as np
import yaml
# from keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import os
import cv2 as cv

config = yaml.load(open(f"./train_classification_config.yaml", "r"))

test_images = os.listdir('./app/test_set/')
predict = []
model = load_model('-02-0.87.h5')

print(test_images)

for i in test_images:
    image = Image.open(f'./app/test_set/{i}')
    image = image.resize((224, 224))
    
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    
    predict.append(np.argmax(model.predict(image)))
print(predict)
# img = preprocess_input('app/test_set/img231.png')


# preds = model.predict(img)


# print(preds)