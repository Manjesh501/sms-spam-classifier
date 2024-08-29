# SMS Spam Classifier

## Overview

The SMS Spam Classifier is a machine learning project that classifies SMS messages as either spam or not spam. The model is trained on a dataset of SMS messages and uses natural language processing (NLP) techniques to predict the nature of new messages.

## Project Structure

- `app.py`: Streamlit application for user interaction and prediction.
- `model_training.py`: Script to train the machine learning model and save it.
- `requirements.txt`: List of Python dependencies.
- `spam.csv`: Dataset used for training the model.
- `vectorizer.pkl`: Pickled TF-IDF vectorizer used for text transformation.
- `model.pkl`: Pickled trained machine learning model.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Manjesh501/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Set Up Virtual Environment**

   ```bash
   python -m venv venv
   ./venv/Scripts/Activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the following command to train the model:

```bash
python model_training.py
```

This will generate the necessary files (`vectorizer.pkl` and `model.pkl`).

### Running the App

To start the Streamlit app, use:

```bash
streamlit run app.py
```

Open the local URL provided in your browser to interact with the application.

## Example

**Input:** "Congratulations! You’ve won a prize."

**Output:** Spam

**Input:** "Are we meeting tomorrow?"

**Output:** Not Spam
