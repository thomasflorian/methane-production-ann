'''
### Import Relevant Libraries
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, help='Input the proportion of the dataset to include in the test split.', default=0.15)
parser.add_argument('--random_state', type=int, help='Input a seed for random selection of the test set.', default=None)
opt = parser.parse_args()
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Input, add
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import time
import innvestigate as inn
from IPython.core.display import HTML

import matplotlib.pyplot as plt

if tf.__version__ != "1.12.0":
    print("WARNING: This demo was designed to use tensorflow version 1.12.0.")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
### Read and Process Data
'''
# Store excel data into DataFrames
xls = pd.ExcelFile("ANN_0530.xlsx")
microbial_data = pd.read_excel(xls, sheet_name = "Microbial Data (Raw Count)")
microbial_data.index = microbial_data.index + 1
digester_data = pd.read_excel(xls, sheet_name = "Digester Methane data", index_col = 0)
microbial_data = microbial_data.iloc[:489,8:].T
# Access pH data
ph_data = digester_data.T[["pH"]]
# Define features and target
X = microbial_data.copy()
Y = digester_data.T[["Methane Production Rate (L-CH4/L-Day)"]]
X.index.name = "Sample"
Y.index.name = "Sample"
# 20% Test/Train Split
trainX, testX, trainY, testY = train_test_split(X,Y, test_size=opt.test_size, random_state=opt.random_state)

'''
### Define and Compile Model
'''
def build_model(input_shape):
    # Define model
    model = keras.models.Sequential([
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=input_shape),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
    ])

    # Compile Model
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    
    return model
print("\nRelevant microbes are being selected with LRP using the training set", end="")

'''
### Select Relevant Features and Train the Model
'''
# Reshape Data for 1D CNN
CNN_trainX = np.asarray(trainX).reshape((trainX.shape[0],trainX.shape[1],1))
CNN_testX = np.asarray(testX).reshape((testX.shape[0],testX.shape[1],1))

# LRP MAGIC
for j in (300, 200, 100):
    # Reset model
    model = build_model((trainX.shape[1],1))
    # Train the model for LRP
    model.fit(CNN_trainX, trainY, batch_size=32, epochs=100, verbose=0)
    # Sort features by LRP Relevance Score
    analyzer = inn.create_analyzer("lrp.z", model)
    lrp = analyzer.analyze(CNN_trainX).reshape((CNN_trainX.shape[0],CNN_trainX.shape[1]))
    lrpDF = pd.DataFrame(lrp.mean(axis=0), index=trainX.columns, columns=["Relevance"])
    lrpDF["Abs(Relevance)"] = np.abs(lrpDF["Relevance"])
    lrpDF.sort_values("Abs(Relevance)",ascending=False, inplace=True)
    lrpDF.drop("Abs(Relevance)", axis=1, inplace=True)
    # Select top features
    trainX = trainX[lrpDF.index[:j]]
    testX = testX[lrpDF.index[:j]]
    # Reshape Data for 1D CNN
    CNN_trainX = np.asarray(trainX).reshape((trainX.shape[0],trainX.shape[1],1))
    CNN_testX = np.asarray(testX).reshape((testX.shape[0],testX.shape[1],1))
    print(".", end="")
print("\nThe model is being trained with", j, "relevant microbes.")
       
# Reset model
model = build_model((trainX.shape[1],1))
# Training the model with relevant features
model.fit(CNN_trainX, trainY, batch_size=32, epochs=100, verbose=1)
# Record model prediction, observed value, error, and r2 score
prediction = model.predict(CNN_testX).flatten()
observed = testY.values.flatten()
error = np.square(prediction - observed).mean()
r2 = r2_score(observed, prediction)

# Print Results
print("\nResults:")
print(pd.DataFrame([prediction,observed], index=["Predicted Methane Production Rate", "True Methane Production Rate"], columns=testY.index).T)
print("R2 Score:",r2)
