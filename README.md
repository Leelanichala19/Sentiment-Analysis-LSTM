# Sentiment Analysis using LSTM (Deep Learning)

## Overview

This project implements a **Sentiment Analysis model** using **Deep Learning (LSTM)** to classify text reviews as **Positive** or **Negative**.
The model is trained on a large-scale dataset and achieves high accuracy, making it suitable for real-world NLP applications.

## Features

* Text preprocessing using **Tokenization & Padding**
* Deep Learning model using **Embedding + LSTM**
* Handles large-scale dataset (3.6 million samples)
* High performance with ~94.8% validation accuracy
* Real-time sentiment prediction function
* Model and tokenizer saved for reuse

## Model Architecture

The model is built using the following layers:

* **Embedding Layer** – Converts words into dense vectors
* **LSTM Layer** – Captures sequential dependencies in text
* **Dropout Layer** – Prevents overfitting
* **Dense Layer** – Learns complex patterns
* **Output Layer (Sigmoid)** – Binary classification (Positive/Negative)

## Dataset

* Dataset: `train.ft.txt` (Amazon Reviews dataset)
* Total samples: **3,600,000**
* Labels:

  * `__label__1` → Negative (0)
  * `__label__2` → Positive (1)

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn

##  Workflow

1. Load dataset from file
2. Extract text and labels
3. Convert labels into binary format
4. Split data into training and validation sets
5. Tokenize text data
6. Pad sequences to fixed length
7. Build LSTM model
8. Train and evaluate the model
9. Save trained model and tokenizer
10. Perform sentiment prediction

## Results

* Training Accuracy: ~95%
* Validation Accuracy: **~94.8%**

## Example Prediction

Input:
"The staff were very helpful and the service was excellent."

Output:
Positive (Confidence: ~0.97)

Input:
"I don't like this product"

Output:
Negative (Confidence: ~0.10)

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook sentiment_analysis.ipynb
```
## Model Saving

The trained model and tokenizer are saved for reuse:

* `sentiment_lstm_model.h5`
* `tokenizer.pkl`

## Future Improvements
* Add Bidirectional LSTM for better context understanding
* Deploy using Streamlit for real-time UI
* Add evaluation metrics (Precision, Recall, F1-score)
* Use pre-trained embeddings like GloVe

##  Author
Leela Nichala Chilakala
GITAM University

## Note
Dataset is not included due to large size. Please download and place it in the appropriate directory before running the notebook.
