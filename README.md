# Predicting anaerobic digester methane production rate with the microbial community composition using deep neural network modeling techniques.

### Abstract
A one-dimensional convolutional neural network (CNN) is trained with raw count data from a microbial community composition dataset to predict the methane production rate of a given sample. The dataset contains data on 489 microbes (features) across 149 sites (samples). This model can outperform linear models that struggle with the high dimensional data and utilize the non-linearity of neural networks to better capture the relationship between the taxa and the methane production rate. 

### Neural Network Architecture
The multilayer convolutional neural network used has six main layers, each consisting of three 1D convolutional layers with a rectified linear unit (ReLU) activation function and a filter size of 1 followed by a dropout layer with a dropout value of 0.1. The number of filters in each of the 1D convolutional layers is 128 in the first two main layers, 64 in the next two main layers, and 32 in the last two main layers. Following the main layers, the model is flattened into a 64 neuron dense layer with a ReLU activation function and then fully connected into a single output. The model was compiled with a mean squared error loss function and an Adam optimizer with a 0.001 learning rate.

  ![Architecture](https://i.imgur.com/0pb54ha.png)

### Feature Reduction Techniques
Having a high dimensional dataset with a relatively low number of samples introduces addition risk of overfitting the model (often referred to as the curse of dimensionality). Layer-wise relevance propagation (LRP) was used to identify the microbes that are important for the model’s prediction by running a backward pass in the neural network. A mean relevance score can be calculated for each microbe, allowing them to be ranked. Taxa with a high magnitude mean relevance score are significant, while taxa with a mean relevance score near zero are less relevant to the model prediction. This score was utilized to select relevant features to use in the training process. Feature reduction techniques using the mean relevance score have proved to produce performance improvements in the model; however, collecting additional data samples could result in a significant improvement of the CNN’s predictive power and reduce the need for feature reduction. Results are compared to a multiple linear regression (MLR) model.

### Validation Techniques
A leave-one-out cross validation of the entire dataset was used to evaluate the performance of the CNN. One sample was removed from the dataset to be used as a validation sample while the rest of the dataset was used as the training data. Feature selection would then occur to select the most relevant features using only the training data in order to protect the validation process from possible bias. This ensures that the ground truth data of the validation sample was not used in any way towards the validation prediction, giving a more accurate representation of how the CNN would perform on predicting the methane production rate of new digestors. After was the model was trained with the feature selected training data, the model would predict the methane production rate of the excluded validation sample. This process was repeated for each of the 149 samples in the dataset, yielding a prediction for each sample. The predictions could then be compared to the ground truth methane production rate and the mean squared error could be calculated to measure performance. 

# Demo
The model demo is available in the demo directory. Running the demo will run a cross validation of the input data with both the CNN and MLR model, where feature reduction uses LRP for the CNN and F-tests for MLR. To run the model demo with default parameters: `python demo.py`. The demo defaults to a leave-one-out cross validation using all features.

Specify the number of folds and number of features for cross validation as integers: `python demo.py --num_folds NUM_FOLDS --num_features NUM_FEATURES`

Specify the path for the feature and ground truth csv data files as strings: `python demo.py --X_path X_PATH --y_path Y_PATH`

Specify the output mode of demo as a boolean: `python demo.py --file_output FILE_OUTPUT`

Due to the stochastic nature of neural networks, the results of the demo will vary. The results of a sample run are shown below:

![Demo Results](demo/demo_plot.png)

# Credits
Author: Thomas Florian @thomasflorian @ [Ye's MLIP Lab]
