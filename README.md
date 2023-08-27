# Sentiment-Analysis-using-Deep-Learning
This document outlines the structure of movie review sentiment analysis model, which aims to predict whether a given review carries a positive or negative sentiment.
This repository contains 2 ipynb files2. 
One is for training model. (train_sentiment_analysis) 
Second is for testing the trained model in unseen review (inference_sentiment_analysis)
Both are jupyter notebook file and model is trained using google colab.

ABOUT MODEL:

This model is built by using Keras, pre-trained GloVe word embeddings and LSTM for sentiment analysis of IMBD movie reviews data set.


Model Artitecture 
LSTM model is built with these layers
Embedding Layer:

LSTM Layer 1: Utilizing 128 memory units with 0.5 dropout

LSTM Layer 2: Utilized 32 memory units, and employs dropout for balance.

Dense Output Layer: Analyzes the gathered data to predict positive/negative sentiment, using the sigmoid acitvation function.

Training Strategy: Optimized by Adam, assessed by Mean Squared Error, and tracked for accuracy for training and validation dataset.

Checkpoints: Model is using checkpoint to saved the progress to ensure the best results.


TESTING:
inference_sentiment_analysis file ask for user input of movie review and predict the sentiment by utilizing trained LSTM model.






