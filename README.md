# Coral Bleaching Image Classifier

Team project — Python, scikit-learn.

## Overview
This project builds a set of machine learning models to classify images of 
coral as bleached or unbleached, using a labeled image dataset sourced from 
Kaggle. The pipeline covers dataset preparation (train/test/validation 
splitting), image preprocessing and feature extraction, and training 
several classification models to compare performance. A GUI allows a user 
to upload their own coral image and get a live classification from the 
trained models.

## Contents
- `model.py` — implements and trains six classification models: 
  Logistic Regression, MLP (Multi-Layer Perceptron), K-Nearest Neighbors, 
  Ridge Classifier, Naive Bayes, and Radius Neighbors Classifier

## My Contribution
I built the classification models (`models.py`), including data splitting, 
feature extraction into model-ready inputs, training, and evaluation across 
all six algorithms. Model accuracy visualization tools and the GUI
were built by a teammates; I adapted my model code's so it integrated cleanly 
with the GUI's expected inputs/outputs.

## Skills Demonstrated
- Supervised classification using multiple scikit-learn algorithms
- Image preprocessing and feature extraction for ML pipelines
- Dataset preparation (train/validation/test splitting)
- Model evaluation and comparison across algorithms
- Collaborative software development — integrating code across a team

## Tools
Python, scikit-learn, skimage, numpy, matplotlib, os, seaborn, pandas.
