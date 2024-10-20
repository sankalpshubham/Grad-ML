# CS 6375 - Graduate Machine Learning <br> Assignment 1
By Sankalp Shubam and Aarian Ahsan

**IMPORTANT:** The file _word_embedding.pkl_ under the Data Embedding folder is not included due to storage limitations. Be sure to download this first before running this script

## Introduction
For this project, we implemented and experimented with two types of neural networks—Feedforward Neural Networks (FFNN) and Recurrent Neural Networks (RNN)—to solve a 5-class sentiment analysis task using Yelp review data. 
The goal of the task is to predict the star rating y∈{1,2,3,4,5} based on the text of the reviews. 
We conducted multiple experiments with different model configurations, focusing on variations in hidden layer dimensions to observe their impact on performance metrics such as training accuracy, validation accuracy, and loss.

Data:
The dataset consists of Yelp reviews paired with star ratings from 1 to 5. The data is split into training, validation, and test sets. 
For the RNN, we utilized a pre-trained word embedding file (word_embedding.pkl) to initialize the input representation for the reviews. 
The exact size of the train, validation, and test sets is reported in the table below.


Results:
In our experiments, we varied the hidden dimensions for both the FFNN and RNN models to evaluate their effect on performance. 
While the FFNN models showed consistent results, with higher validation accuracy for smaller hidden dimensions, the RNN models exhibited convergence issues and lower overall validation accuracy. 
Overall, the FFNN models demonstrated more stability and better generalization than the RNN models.

Tasks & Data Statistics:
The task is a multiclass classification problem where models must classify Yelp reviews into one of five-star ratings (1-5). The input is the text of the review, and the output is the predicted star rating. 
For FFNN, the text is converted into a bag-of-words representation, while for the RNN, pre-trained word embeddings are used to represent the text input.

| Dataset      | Number of Examples |
| :---        |    :----:   |
| Training      | 50,000       |
| Validation   | 10,00        |
| Test   | 10,00        |



## Instructions
**How to run:** (Paste in the terminal when in root folder)

FFNN: `python ffnn.py --hidden_dim 128 --epochs 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json`
<br> 
<br>
RNN: `python rnn.py --hidden_dim 128 --epochs 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json`
