# Predicting anaerobic digester methane production rate with the microbial community composition using deep neural network modelling techniques.

A one-dimensional convolutional neural network (CNN) is trained with raw count data from a microbial community composition dataset to predict the methane production rate of a given sample. The dataset contains data on 489 microbes (features) across 149 sites (samples). Layer-wise relevance propagation (LRP) was used to identify the microbes that are important for the model’s prediction by running a backward pass in the neural network. A mean relevance score can be calculated for each microbe, allowing them to be ranked. Feature reduction techinques using the mean relevance score have proved to produce performance improvements in the model; however, collecting additional data samples could result in a significant improvement of the CNN’s predictive power and reduce the need for feature reduction.

# Demo
To run the model demo with default parameters: `python demo.py `
with the ANN_0530.xlsx file located in the same directory.

Specify test size and random seed: `demo.py --test_size TEST_SIZE --random_state RANDOM_STATE`

# Credits
Author: Thomas Florian @thomasflorian @ [Ye's MLIP Lab]
