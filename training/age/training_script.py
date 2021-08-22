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
DATA_DIR = Path("/home/gabriel-milan/data/age")
X_PATH = DATA_DIR / "X_age.npy"
Y_PATH = DATA_DIR / "y_age.npy"
N_FOLDS = {{n_folds}}
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


# Read X and y
X = np.load(X_PATH)
y = np.load(Y_PATH)

# Setup stratified k-fold
skf = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)

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
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[val_index]
    y_val = y[val_index]

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
        X_train, y_train,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        verbose=VERBOSE
    )

    # Store training and validation scores
    training_scores.append(history.history["auc"][-1])
    validation_scores.append(history.history["val_auc"][-1])

# Get scores average
training_score = np.average(training_scores)
validation_score = np.average(validation_scores)

# Save run
save_run({{run_config}}, model, training_score,
         validation_score, ["age", "age-gender-detection"])
