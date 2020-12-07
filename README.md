# Sentiment Analysis on Amazon Foods
This repository contains code for training and evaluation of the Bidirectional Encoder with Fully Connected layers (BEFC) model for the Sentiment Analysis task on Amazon Fine Foods dataset. The results are compared to SOTA based on Universal Sentence Encoder by Google. Two different hyperparameters optimization engines, BOHB and Optuna, are compared. Please see full report amazon_foods_sentiment_analysis_and_automl.pdf for details.

## Results

|   model  | automl | n_epochs | training time | accuracy |
|----------|--------|----------|---------------|----------|
| USE SOTA |        |          | 5h            | 0.9050   |
| BEFC     | BOHB   | 7        | 5h            | 0.8419   |
| BEFC     | BOHB   | 12       | 8h            | 0.8605   |
| BEFC     | BOHB   | 20       | 10h           | 0.8435   |
| BEFC     | Optuna | 7        | 4.5h          | 0.8877   |
| BEFC     | Optuna | 12       | 8.5h          | 0.8950   |
| BEFC     | Optuna | 20       | 14h           | 0.8908   |


## Architectures
* Bidirectional Encoder
* Fully Connected

## Dependencies
* torch
* Optuna
* BOHB
