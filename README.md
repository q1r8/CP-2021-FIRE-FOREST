# Fire-vision

## Реализованный функционал:
  1. Обучение классификатора (сверточная сеть) изображений
  2. Сервис классификации видео, как онлайн потока, так и записи

## Киллер фичи:
  1. Реализована функция посчета расстояния от камеры до предполагаемого очага возгарания (app/api_cv.py)
  2. Готовый пайплайн для дообучения или переобучения классификатора (src/train.py)
  3. Готовый скрипт для применения основных аугментаций для данных (augmentation.py)

## Основной стек технологий:
  1. tensorflow
  2. keras
  3. opencv
 
## Демо проекта:
https://colab.research.google.com/drive/1Xls56isBRmYn3FrqYvypqzbg1U6gJbTc?usp=sharing

# Среда запуска
 1. Сервис разворачивается на Ubuntu 20.04
 2. Python 3.6.9
 3. virtualenv - виртуальное окружение питона

# Установка
  1. git clone https://github.com/q1r8/CP-2021-FIRE-FOREST.git
  2. python3 -m virtualenv venv
  3. source venv/bin/activate
  4. cd CP-2021-FIRE-FOREST
  5. pip install -r requirements.txt

# Запуск обучения
Проект запускается из корневой папки. Данные разбитые по классам = директориям лежат в одной директории для датасета
dataset/
  -class_1/
    -image1
    -image2
  -class_2/
    -image1
    -image2
    
  1. Поставить нужный путь датасета в конфиге
  2. python src/train.py

# Разработчики
Егор Морозов - tg @egor_moroz
Сергей Башков - tg @https://t.me/bashkovs
