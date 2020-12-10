'''
### Parse Arguments
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_folds', type=int, help='Input the number of leave-one-out cross validation folds.', default=149)
opt = parser.parse_args()

'''
### Import Relevant Libraries
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Silence warnings due to version mismatch. Written with Tensorflow Version == 1.12.0.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
### Read and Process Data
'''

# Load in raw count data for neural network
X = pd.read_csv("Demo_Data/X.csv", index_col=0)
# Calculate relative abundance for linear regression
X_rel = (X.T/X.sum(axis=1)).T
# Load in ground truth methane production rate data
y = pd.read_csv("Demo_Data/Y.csv", index_col=0)
# Load in LRP selected features for each sample
lrp = pd.read_csv("Demo_Data/10_features_lrp.csv", index_col=0, dtype=str)

'''
### Define and Compile Model
'''

def build_model(input_shape):
    # Define model
    model = keras.models.Sequential([
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=input_shape),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
            keras.layers.Dropout(0.1),
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

'''
### Run Linear Regression Cross Validation
'''
mlr_predictions = np.zeros(opt.num_folds)
for fold in range(opt.num_folds):
    # Select sample for validation
    sample = X_rel.iloc[fold].name
    
    # Seperate validation sample from training samples
    X_train = X_rel.drop(sample)
    X_val = X_rel.loc[[sample]]
    y_train = y.drop(sample).values.flatten()
    
    # Select 10 most important features
    with np.errstate(invalid='ignore', divide='ignore'): # Ignore warnings
        f_scores = pd.DataFrame(f_regression(X_train, y_train)[0], index=X_train.columns, columns=["F Score"])
    f_scores.sort_values("F Score", ascending=False, inplace=True)
    X_train = X_train[f_scores.index[:10]]
    X_val = X_val[f_scores.index[:10]]
    
    # Train linear regression model
    regr = LinearRegression().fit(X_train, y_train)
    # Cache prediction values to array
    mlr_predictions[fold] = regr.predict(X_val)

'''
### Run Neural Network Cross Validation
'''
print("\nRunning Leave-one-out Cross Validation:\n")
    
ann_predictions = np.zeros(opt.num_folds)
for fold in range(opt.num_folds):
    # Reset keras session to reduce model clutter
    keras.backend.clear_session()
    
    # Select sample for validation
    sample = X.iloc[fold].name
    
    # Seperate validation sample from training samples
    X_train = X.drop(sample)
    X_val = X.loc[[sample]]
    y_train = y.drop(sample).values.flatten()
    
    # Select 10 most important features
    X_train = X_train[lrp[sample]]
    X_val = X_val[lrp[sample]]
    
    # Reshape data for nerual network
    X_train = np.asarray(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
    X_val = np.asarray(X_val).reshape((X_val.shape[0],X_val.shape[1],1))
    
    # Run neural network model
    model = build_model((X_train.shape[1],1))
    model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
    ann_predictions[fold] = model.predict(X_val)[0]
    
    # Print status update
    print("--------[{}/{}]--------".format(fold+1, opt.num_folds))
    print("Sample:", sample)
    print("ANN Prediction: {:.5f}".format(ann_predictions[fold]))
    print("MLR Prediction: {:.5f}".format(mlr_predictions[fold]))
    print("Ground Truth: {:.5f}".format(y.loc[sample][0]))
    
# Print results    
print("\nCross Validation Results:\n")
ann_r2 = r2_score(y[:opt.num_folds], ann_predictions)
mlr_r2 = r2_score(y[:opt.num_folds], mlr_predictions)
print("Neural Network R2 Score: {:.5f}".format(ann_r2))
print("Linear Regression R2 Score: {:.5f}\n".format(mlr_r2))
