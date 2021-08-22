import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def get_model() -> Model:
    input = layers.Input(shape=(227, 227, 3))
    oup = layers.Conv2D(filters=96, kernel_size=11, strides=4,
                        padding='valid', activation='relu')(input)
    oup = layers.MaxPool2D(pool_size=3, strides=2)(oup)
    oup = layers.BatchNormalization()(oup)
    oup = layers.Conv2D(filters=256, kernel_size=5, strides=1,
                        padding='same', activation='relu')(oup)
    oup = layers.MaxPool2D(pool_size=3, strides=2)(oup)
    oup = layers.BatchNormalization()(oup)
    oup = layers.Conv2D(filters=384, kernel_size=3, strides=1,
                        padding='same', activation='relu')(oup)
    oup = layers.MaxPool2D(pool_size=3, strides=2)(oup)
    oup = layers.Flatten()(oup)
    oup = layers.Dense(4096, activation='relu')(oup)
    oup = layers.Dropout(0.5)(oup)
    oup = layers.Dense(512, activation='relu')(oup)
    oup = layers.Dropout(0.5)(oup)
    oup = layers.Dense(64, activation='relu')(oup)
    oup = layers.Dropout(0.5)(oup)
    oup = layers.Dense(23, activation='softmax')(oup)
    model = Model(inputs=input, outputs=oup)
    return model


if __name__ == '__main__':
    model = get_model()
    with open("model_config.json", "w") as f:
        f.write(model.to_json())
