'''
### Parse Arguments
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_folds', type=int, help='Input the number of folds for cross validation.', default=None)
parser.add_argument('--num_features', type=int, help='Input the number features to be selected.', default=None)
parser.add_argument('--X_path', type=str, help='Input the path to the features data csv file.', default="X.csv")
parser.add_argument('--y_path', type=str, help='Input the path to the features data csv file.', default="y.csv")
parser.add_argument('--file_output', type=bool, help='Output to file', default=False)

opt = parser.parse_args()

'''
### Import Relevant Libraries
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import innvestigate as inn
import os
import sys


# Silence warnings due to version mismatch. Written with Tensorflow Version == 1.12.0.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if tf.__version__[0] != "1.12.0":
    print("Warning: This code was developed with Tensorflow Version 1.12.0. You're running Tensorflow Version {}.".format(tf.__version__))

# Select output
if opt.file_output:
    orig_stdout = sys.stdout
    f = open('demo_results.txt', 'w')
    sys.stdout = f

'''
### Read and Process Data
'''
# Load in raw count data for neural network
X = pd.read_csv(opt.X_path, index_col=0)
# Calculate relative abundance for linear regression
X_rel = (X.T/X.sum(axis=1)).T
# Load in ground truth methane production rate data
y = pd.read_csv(opt.y_path, index_col=0)
# Set parameters
num_samples = X.shape[0]
num_folds = X.shape[0] if opt.num_folds == None else opt.num_folds
num_features = X.shape[1] if opt.num_features == None else opt.num_features
# Create linearly spaced chunks for cross validation
chunks = np.ceil(np.linspace(0,num_samples, num=num_folds+1)).astype(int)

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


print("\nThe loaded dataset has {} samples and {} features, and".format(num_samples, X.shape[1]))
print("{} features will be selected in the {}-fold cross validation.".format(num_features, num_folds))
print("\n... Running cross validation:\n")


'''
### Run Linear Regression Cross Validation
'''
mlr_predictions = np.array([])
for fold in range(num_folds):

    # Select validation samples
    X_val = X_rel[chunks[fold]:chunks[fold+1]]
    # Select training samples 
    X_train = X_rel.drop(X_val.index)
    y_train = y.drop(X_val.index).values.flatten()

    # Select most important features
    with np.errstate(invalid='ignore', divide='ignore'): # Ignore warnings
        f_scores = pd.DataFrame(f_regression(X_train, y_train)[0], index=X_train.columns, columns=["F Score"])
    f_scores.sort_values("F Score", ascending=False, inplace=True)
    X_train = X_train[f_scores.index[:num_features]]
    X_val = X_val[f_scores.index[:num_features]]

    # Train linear regression model
    regr = LinearRegression().fit(X_train, y_train)
    # Cache prediction values to array
    predictions = regr.predict(X_val).flatten()
    mlr_predictions = np.concatenate([mlr_predictions, predictions])


'''
### Run Neural Network Cross Validation
'''

ann_predictions = np.array([])
for fold in range(num_folds):

    # Reset keras session to reduce model clutter
    keras.backend.clear_session()

    # Select validation samples
    X_val = X[chunks[fold]:chunks[fold+1]]
    # Select training samples 
    X_train = X.drop(X_val.index)
    y_train = y.drop(X_val.index).values.flatten()

    # Feature selection using Layerwise Relevance Propegation (LRP)
    model = build_model((X_train.shape[1],1))
    model.fit(np.expand_dims(X_train.values, axis=2), y_train, batch_size=32, epochs=150, verbose=0)
    # Sort features by LRP Relevance Score
    analyzer = inn.create_analyzer("lrp.z_plus_fast", model)
    # Perform backwards pass through trained neural network to generate relevance scores
    scores = analyzer.analyze(np.expand_dims(X_train.values, axis=2))[...,0]
    # Store data in Dataframe
    lrp = pd.DataFrame(scores.mean(axis=0), index=X_train.columns, columns=["Score"])
    # Sort scores by absolute value
    lrp["Abs Score"] = np.abs(lrp["Score"])
    lrp.sort_values(by="Abs Score", ascending=False, inplace=True)

    # Select most important features
    X_train = X_train[lrp.index[:num_features]]
    X_val = X_val[lrp.index[:num_features]]

    # Reshape data for nerual network
    X_train = np.asarray(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
    X_val = np.asarray(X_val).reshape((X_val.shape[0],X_val.shape[1],1))

    # Run neural network model
    model = build_model((X_train.shape[1],1))
    model.fit(X_train, y_train, batch_size=32, epochs=150, verbose=0)

    # Cache prediction values to array
    predictions = model.predict(X_val).flatten()
    ann_predictions = np.concatenate([ann_predictions, predictions])

    # Print status update
    print("--------[{}/{}]--------".format(fold+1, num_folds))
    for i in range(chunks[fold+1] - chunks[fold]):
        print("Validation Sample:", y.index.values[chunks[fold]+i])
        print("ANN Prediction: {:.5f}".format(ann_predictions[chunks[fold]+i]))
        print("MLR Prediction: {:.5f}".format(mlr_predictions[chunks[fold]+i]))
        print("Ground Truth: {:.5f}\n".format(y.values.flatten()[chunks[fold]+i]))

# Print results    
print("\nCross Validation Results:\n")
ann_r2 = r2_score(y, ann_predictions)
mlr_r2 = r2_score(y, mlr_predictions)
ann_mse = np.mean(np.square(y.values.flatten() - ann_predictions))
mlr_mse = np.mean(np.square(y.values.flatten() - mlr_predictions))
print("Neural Network R2 Score: {:.5f}".format(ann_r2))
print("Linear Regression R2 Score: {:.5f}".format(mlr_r2))
print("Neural Network MSE: {:.5f}".format(ann_mse))
print("Linear Regression MSE: {:.5f}".format(mlr_mse))

# Store cross validation results in DataFrame
cv_results = pd.DataFrame([ann_predictions, mlr_predictions, y.values.flatten()], index=["ANN", "MLR", "TRUE"], columns=X.index).T
# Calculate percent error for each prediction
cv_results["ANN ERROR"] = np.multiply(np.abs(np.divide(cv_results["ANN"] - cv_results["TRUE"], cv_results["TRUE"])),100)
cv_results["MLR ERROR"] = np.multiply(np.abs(np.divide(cv_results["MLR"] - cv_results["TRUE"], cv_results["TRUE"])),100)
# Quantize percent error to use as color in plot
# <10% = 0, 10%-25% = 1, >25% = 2
cv_results["ANN HUE"] = 0
cv_results["MLR HUE"] = 0
cv_results.loc[(cv_results["ANN ERROR"] > 10) & (cv_results["ANN ERROR"] < 25), "ANN HUE"] = 1
cv_results.loc[cv_results["ANN ERROR"] >= 25, "ANN HUE"] = 2
cv_results.loc[(cv_results["MLR ERROR"] > 10) & (cv_results["MLR ERROR"] < 25), "MLR HUE"] = 1
cv_results.loc[cv_results["MLR ERROR"] >= 25, "MLR HUE"] = 2

# Plot ANN results
fig,(ax1,ax2) = plt.subplots(1,2, sharey=True, sharex=True, figsize=(12,8), dpi=100, linewidth=2, edgecolor="k")
g1 = sns.scatterplot(ax=ax1, data=cv_results, x="TRUE", y="ANN", s=80, hue="ANN HUE", palette={0:"limegreen", 1:"orange", 2:"red"})
g2 = sns.scatterplot(ax=ax2, data=cv_results, x="TRUE", y="MLR", s=80, hue="MLR HUE", palette={0:"limegreen", 1:"orange", 2:"red"})
g1.add_line(plt.Line2D([0,1],[0,1], color="black", lw=1));
g2.add_line(plt.Line2D([0,1],[0,1], color="black", lw=1));
g1.set_ylim(0,1); g1.set_xlim(0,1);
g2.set_ylim(0,1); g2.set_xlim(0,1);
g1.annotate(r"MSE = {:.4f}".format(ann_mse), (0.5,0.91), fontsize=15);
g2.annotate(r"MSE = {:.4f}".format(mlr_mse), (0.5,0.91), fontsize=15);
g1.legend(loc='center left', bbox_to_anchor=(0.08, 1.03), fontsize=8, ncol=3, handles = g1.get_legend_handles_labels()[0][1:], labels=("<10% error", "10%-25% error", ">25% error"), handletextpad = 0, frameon=False);
g2.legend(loc='center left', bbox_to_anchor=(0.08, 1.03), fontsize=8, ncol=3, handles = g2.get_legend_handles_labels()[0][1:], labels=("<10% error", "10%-25% error", ">25% error"), handletextpad = 0, frameon=False);
g1.set_xlabel(r"Actual Methane Production Rate (L–CH$_4$/L$ _R$–Day)", fontsize=11)
g2.set_xlabel(r"Actual Methane Production Rate (L–CH$_4$/L$ _R$–Day)", fontsize=11)
g1.set_ylabel(r"Predicted Methane Production Rate (L–CH$_4$/L$ _R$–Day)", fontsize=11)
g1.set_title("{} Feature Deep Neural Network".format(num_features), fontsize=13, pad=30)
g2.set_title("{} Feature Multiple Linear Regression".format(num_features), fontsize=13, pad=30)
g1.grid(); g2.grid();
fig.savefig("demo_plot", edgecolor=fig.get_edgecolor());
plt.show();

# Reset output
if opt.file_output:
    sys.stdout = orig_stdout
    f.close()
