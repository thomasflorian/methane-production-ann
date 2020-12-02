# Predicting anaerobic digester methane production rate with the microbial community composition using deep neural network modelling techniques.

A one-dimensional convolutional neural network (CNN) is trained with raw count data from a microbial community composition dataset to predict the methane production rate of a given sample. The dataset contains data on 489 microbes (features) across 149 sites (samples). Layer-wise relevance propagation (LRP) was used to identify the microbes that are important for the model’s prediction by running a backward pass in the neural network. A mean relevance score can be calculated for each microbe, allowing them to be ranked. Feature reduction techinques using the mean relevance score have proved to produce performance improvements in the model; however, collecting additional data samples could result in a significant improvement of the CNN’s predictive power and reduce the need for feature reduction.

# Demo
The model demo is avaliable in the demo directory. Running the demo will randomly select training and test samples from the digester dataset, use LRP to select the relevant features from the training set, and train the model using those relevant features to predict the methane production rate of the test samples. Due to the random nature of the test/train split, the results of the demo will vary. A random seed can be specified with the --random_state flag. Due to the limited number of samples in the dataset, using a large test set will impact the performance of the model as the number of training samples is limited. The test size can be specified with the --test_size flag, 0.15 by default.

To run the model demo with default parameters: `python demo.py` with the ANN_0530.xlsx file located in the same directory.

Specify test size and random seed: `python demo.py --test_size TEST_SIZE --random_state RANDOM_STATE`

# Credits
Author: Thomas Florian @thomasflorian @ [Ye's MLIP Lab]
