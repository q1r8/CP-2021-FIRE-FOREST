import albumentations as A
import cv2
import os
from tqdm.auto import tqdm

# Declare an augmentation pipeline
transform = A.Compose([
    # A.HorizontalFlip(p=1),
    # A.Rotate(limit=12, p=1)
    # A.RandomResizedCrop(width=1280, height=720, scale=(0.85, 0.85), p=1)
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=1),
    # A.RandomCrop(width=1200, height=675, p=0.5),
    # A.RandomFog()
    # A.Crop(p=1.0, x_min=0, y_min=130, x_max=1280, y_max=720),
    # A.RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(1, 1), p=1),
    # A.InvertImg(always_apply=False, p=1.0)
])

path = '' # data path for augmentation
for img in tqdm(sorted(os.listdir(path))):
    if img.endswith('.png'):
        image = cv2.imread(f"{path+img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        new_dataset_path = ''
        cv2.imwrite(f"{path}/{img.split(sep='.')[0]}.png", transformed_image)