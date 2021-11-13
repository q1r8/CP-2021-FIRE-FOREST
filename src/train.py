import yaml
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras import layers
from keras.applications import MobileNetV2


def make_dataset(config):
    data_generator = ImageDataGenerator(
        validation_split=config.get('validation_split'),
        horizontal_flip=True,
        rotation_range=15
    )

    train_data_loader = data_generator.flow_from_directory(
        directory=config.get('dataset_path'),
        target_size=(config.get('img_width'), config.get('img_height')),
        subset='training'
    )

    val_data_loader = data_generator.flow_from_directory(
        directory=config.get('dataset_path'),
        target_size=(config.get('img_width'), config.get('img_height')),
        subset='validation'
    )
    return train_data_loader, val_data_loader


def make_model_backbone():
    base_resnet = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(config.get('img_width'), config.get('img_height'), 3))

    model = Sequential()
    model.add(base_resnet)
    model.add(layers.Dense(2, activation='softmax'))
    return model


def train_model(model, train_generator, val_generator):
    model.compile(loss='categorical_crossentropy',
                       optimizer=Adam(lr=config.get('learning_rate'), decay=1e-6),
                       metrics=['accuracy'])

    rlr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=2, factor=0.7)
    checkpoint = ModelCheckpoint(filepath='-{epoch:02d}-{val_accuracy:.2f}.h5',
                                 monitor='val_accuracy', verbose=2, save_best_only=True)

    model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=config.get('epochs'),
                    callbacks=[rlr, checkpoint])


if __name__=='__main__':
    config = yaml.load(open(f"./train_classification_config.yaml", "r"))
    train_data_generator, val_data_generator = make_dataset(config)
    model = make_model_backbone()
    train_model(model, train_data_generator, val_data_generator)
