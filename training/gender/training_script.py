from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from coach.db import save_run
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATA_DIR = Path("/home/gabriel-milan/data/gender")
RANDOM_SEED = {{random_seed}}
MAX_EPOCHS = {{max_epochs}}
LOSS = {{loss}}
OPTIMIZER = {{optimizer}}
VERBOSE = {{verbose}}
ES_PATIENCE = {{es_patience}}
LR_PATIENCE = {{lr_patience}}
LR_FACTOR = {{lr_factor}}


def get_model(model_config=None) -> Model:
    """
    Function for getting model based on JSON config
    """
    if not model_config:
        with open("model_config.json", "r") as f:
            model_config = f.read()
    model = tf.keras.models.model_from_json(model_config)
    return model


# Read dataframe with filenames and labels
data_df = pd.read_csv(str(DATA_DIR / "gender_dataset.csv"))
y = data_df["gender"]
data_df["gender"] = data_df["gender"].astype(str)

# Setup stratified k-fold
skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

# Setup an image data generator for providing data
idg = ImageDataGenerator()

# Store scores while training
validation_scores = []
training_scores = []

# Iterate over folds, getting train and validation indices
for train_index, val_index in skf.split(np.zeros(len(y)), y):

    # Clear session
    K.clear_session()

    # Get train and validation data
    training_data = data_df.iloc[train_index]
    validation_data = data_df.iloc[val_index]

    # Set generatorss
    train_data_generator = idg.flow_from_dataframe(
        dataframe=training_data,
        x_col="image_name",
        y_col="gender",
        class_mode="binary",
        shuffle=True
    )
    validation_data_generator = idg.flow_from_dataframe(
        dataframe=validation_data,
        x_col="image_name",
        y_col="gender",
        class_mode="binary",
        shuffle=True
    )

    # Get model and compile
    model = get_model()
    model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER,
        metrics=["AUC"]
    )

    # Setup callbacks
    lrr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=VERBOSE,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=ES_PATIENCE,
        verbose=VERBOSE,
        mode="auto"
    )

    callbacks = [lrr, es]

    # Train model
    history = model.fit(
        train_data_generator,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        validation_data=validation_data_generator,
        verbose=VERBOSE
    )

    # Store training and validation scores
    training_scores.append(history.history["auc"])
    validation_scores.append(history.history["val_auc"])

# Get scores average
training_score = np.average(training_scores)
validation_score = np.average(validation_scores)

# Save run
save_run({{run_config}}, model, training_score,
         validation_score, ["gender", "age-gender-detection"])
