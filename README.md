# ML-project
This project involves implementing a decision tree classifier from scratch to predict whether mushrooms are edible or poisonous using the Mushroom dataset. The project includes creating tree nodes, training the model with various criteria, evaluating its performance, and optimizing hyperparameters to improve accuracy and prevent overfitting.
# Mushroom Classification with Decision Trees

## Project Overview
This project implements a decision tree classifier from scratch to predict whether mushrooms are edible or poisonous based on their physical characteristics. The Mushroom dataset, which contains categorical features, is used for training and evaluation.

## Features
- **Custom Decision Tree Implementation:** A decision tree classifier is built from scratch using Python.
- **Node Class:** Represents each node in the decision tree, handling splits and storing decision criteria.
- **Tree Building and Splitting:** Implements Gini index and entropy-based splitting criteria.
- **Hyperparameter Tuning:** Optimizes tree depth and minimum samples per split to improve performance.
- **Evaluation:** Assesses model accuracy on training and test datasets, with a focus on mitigating overfitting.

## Dataset
- **Source:** Mushroom Dataset
- **Description:** The dataset includes 22 categorical features, with the target variable indicating whether a mushroom is edible (0) or poisonous (1).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mushroom-classification.git
   cd mushroom-classification
