# scikit-learn Update: TF-IDF + SVM Model

This repository contains an updated implementation of a text classification model using Scikit-learn's `TfidfVectorizer` and `LinearSVC`. The goal is to ensure compatibility with the latest version of Scikit-learn.

## Features
- **TF-IDF Vectorization**: Transforms text data into numerical vectors using the `TfidfVectorizer`.
- **SVM Classifier**: Utilizes the `LinearSVC` model for classification tasks.
- **Pipeline**: Combines the vectorizer and classifier into a seamless pipeline.
- **Model Persistence**: Saves the trained model and vectorizer for later use.
- **Dataset**: Uses the `20 Newsgroups` dataset from Scikit-learn for training.

## Installation

To use this repository, make sure you have the latest version of Scikit-learn installed:

```bash
pip install scikit-learn --upgrade
