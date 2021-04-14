import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from tensorflow import keras
from Forecasters.TSDatasetGenerator import TSDatasetGenerator


## Auxiliary for classification report ##
def genClassificationRep(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))


# Training and Testing Data
train_data = pd.read_csv('rec_train.csv').drop(columns=['Unnamed: 0', 'DATE'])
test_data = pd.read_csv('rec_test.csv').drop(columns=['Unnamed: 0', 'DATE'])

# Generate h step ahead forecast dataset with 1 lag of Y and 1 lag of X
dg = TSDatasetGenerator()
Dataset = dg.fit_transform(train_data, 'Is_Recession', h=12, l=1, k=1)

# Split training data into X and y
X_train = Dataset.drop(columns=['Is_Recession']).astype('float32')
y_train = Dataset['Is_Recession']

# Standardise data
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)

# Configs
BATCH_SIZE = 1  # 1 batch
WINDOW = 100  # 100 months look back

## Dataset Creation ##
# Create TS dataset for Keras
ts_dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_train_std,
    y_train,
    sequence_length=WINDOW,  # Size of sliding window (~4 year window = 100)
    batch_size=BATCH_SIZE
)

# Get dimensions of dataset
for batch in ts_dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

# Create Test TS dataset for Keras
Dataset_Test = dg.fit_transform(test_data, 'Is_Recession', h=1, l=1, k=1)
Dataset_Test.head()
# Split training data into X and y
X_test = Dataset_Test.drop(columns=['Is_Recession']).astype('float32')
y_test = Dataset_Test['Is_Recession']

# Standardise data
X_test_std = std_scaler.fit_transform(X_test)

ts_dataset_test = keras.preprocessing.timeseries_dataset_from_array(
    X_test,
    y_test,
    sequence_length=WINDOW,  # Size of sliding window (~4 year window = 100)
    batch_size=BATCH_SIZE
)

# Get dimensions of dataset
for batch in ts_dataset_test.take(1):
    test_inputs, test_targets = batch

print("Input shape:", test_inputs.numpy().shape)
print("Target shape:", test_targets.numpy().shape)

## Simple RNN model ##
# Add in regularisation (Elastic net)
elnet_reg = keras.regularizers.L1L2(0.01, 0.01)

# Compile model
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
rnn_hidden1 = keras.layers.SimpleRNN(32, input_shape=[None, 38], recurrent_regularizer=elnet_reg)(inputs)
outputs = keras.layers.Dense(1, activation='sigmoid')(rnn_hidden1)

rnn_model = keras.Model(inputs=inputs, outputs=outputs)
rnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy()])

rnn_model.summary()

# Plot model
tf.keras.utils.plot_model(
    rnn_model,
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    dpi=300,
)

# Callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

# Fit Model
rnn_history = rnn_model.fit(
    ts_dataset_train,
    epochs=100,
    validation_data=ts_dataset_test,
    callbacks=[early_stopping_cb]
)

# Visualise loss
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(rnn_history, "RNN: Training and Validation Loss")

# Prediction
def getPredictions(model, tf_df):
    y_pred_proba = []
    y_pred = []
    y_val = []
    for x, y in tf_df:
        pred = rnn_model.predict(x)
        y_pred_proba.append(pred)
        y_pred.append(1 if pred > 0.5 else 0)
        y_val.append(y)
    return (y_pred, y_val, y_pred_proba)


# Get predictions
rnn_y_pred, rnn_y_true, rnn_y_pred_proba = getPredictions(rnn_model, ts_dataset_test)

# Eval RNN model
genClassificationRep(rnn_y_true, rnn_y_pred)
confusion_matrix(rnn_y_true, rnn_y_pred)
rnn_log_loss = log_loss([i.numpy()[0] for i in rnn_y_true], rnn_y_pred)

## ARCHIVE ##
# stacked RNN (Effect of low signal to noise ratio more pronounced)
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
srnn_hidden1 = keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 38],
                                      recurrent_regularizer=elnet_reg)(inputs)
srnn_hidden2 = keras.layers.SimpleRNN(32, return_sequences=True, recurrent_regularizer=elnet_reg)(srnn_hidden1)
outputs = keras.layers.Dense(1, activation='sigmoid')(srnn_hidden2)

srnn_model = keras.Model(inputs=inputs, outputs=outputs)
srnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   loss="binary_crossentropy",
                   metrics=[keras.metrics.BinaryAccuracy()])

srnn_model.summary()

# LSTM (Trying memory cells -> Similar results but more expensive to train)
# Try layer norm in case of exploding/ vanishing gradients
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32, input_shape=[None, 38], recurrent_regularizer=elnet_reg)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

lstm_model = keras.Model(inputs=inputs, outputs=outputs)
lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   loss="binary_crossentropy",
                   metrics=[keras.metrics.BinaryAccuracy()])
lstm_model.summary()
