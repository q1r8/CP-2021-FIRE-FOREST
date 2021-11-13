import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from keras.models import load_model

def preprocess_input_data(img):
    image = Image.open(img)
    image = image.resize((224, 224))

    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)

    return image


def get_model(path):
    return load_model(path)


def get_photo_annotation(img, fire_presence=f'FIRE DETECTED', position=(10, 50), color=(0, 0, 255, 0)):
    image_text = fire_presence + datetime.now().strftime("%h-%m-%s")
    return cv2.putText(img, image_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


def distance(height_tree_px, focal_lenght=0.00444, tree_height=37):
    height_object_px = height_tree_px
    image_size = [1280, 720]  # пиксели
    matrix_size = [5.1, 3.8]  # миллиметры

    focus = focal_lenght
    height_object = tree_height
    height_on_matrix = (matrix_size[0] / 1000) / image_size[0] * height_object_px

    distance = (focus * height_object) / height_on_matrix
    return round(distance)


def video_analyze(video_path, model_path):
    model = get_model(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Cannot to open video file')

    while cap.isOpened():
        fl, img_frame = cap.read()
        if img_frame is None:
            break

        image = Image.fromarray(img_frame.astype('uint8'), 'RGB').resize((320, 320))

        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)

        model_predicts = np.argmax(model.predict(image))

        if model_predicts == 0:
            get_photo_annotation(image)
            print(image.shape)

        distance_to_fire = distance(14) # количество пикселей, занимаемое дымом или огнем на изображении
        print(distance_to_fire)
        '''
        Сделать реализацию какого то PUSH уведомления, если модель отвечает нулем.
        Если получить больше данных, можно обучить сегментатор, с помощью которого можно наиболее точно определять
        расстояние до пожара и примерную площадь очага возгарания.
        '''


if __name__=='__main__':
    video_analyze(video_path='', model_path='')