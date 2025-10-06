# Spam Email Classification

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text_Classification-green.svg)
![F1 Score](https://img.shields.io/badge/F1-92%25-success)

## Overview
This project implements a deep learning solution for classifying emails as spam or non-spam. Using a combination of **LSTM and CNN**, the model achieves a **F1-score of 92%** on a completely **independent test set of 33,000 emails**. The pipeline includes full text preprocessing, tokenization, and vectorization for optimal model performance.

### Key Achievements
- Preprocessed and cleaned email data  
- Tokenization and vectorization of text for deep learning input  
- Implemented hybrid **LSTM + CNN** architecture for sequence modeling and feature extraction  
- Achieved **F1-score of 92%** on **33,000 completely unseen emails**  

## Dataset
- **Source**: Public spam/non-spam email dataset (e.g., Enron Spam Dataset) for training/validation  
- **Size**: 33,000 emails used as **independent test set** (brand new, never seen by the model)  
- **Split**: Original dataset split into 75% training / 25% validation  

## Requirements
- Python 3.6+  
- TensorFlow 2.x  
- NumPy  
- Pandas  
- Matplotlib  
- NLTK / SpaCy (for text preprocessing)  

## Model Architecture
Text Input  
  ├── Tokenization + Embedding  
  ├── LSTM Layer (sequence modeling)  
  ├── CNN Layer (feature extraction)  
  ├── GlobalMaxPooling1D  
  └── Dense Layer with Sigmoid (spam/non-spam)  

### Model Performance
| Metric    | Training/Validation | Independent Test Set (33k emails) |
|-----------|-------------------|---------------------------------|
| Accuracy  | 98%               | 91%                             |
| Precision | 98%               | 92%                             |
| Recall    | 98%               | 91%                             |
| F1-Score  | 98%               | 92%                             |

## Quick Start
1. Clone this repository:

    git clone https://github.com/yourusername/spam-email-classification.git  
    cd spam-email-classification

2. Install dependencies:

    pip install -r requirements.txt

3. Place your dataset in the `data/` directory  

4. Run training:

    python train.py

5. Run predictions:

    python predict.py --email "Your email text here"

## Project Structure
data/  
  ├── train/  
  └── test/  
notebooks/  
  └── Spam_Classification_Final.ipynb  
src/  
  ├── train.py  
  └── predict.py  
README.md  

## Future Development
- Integration of **BERT or Transformer-based models** for improved accuracy  
- Real-time email filtering  
- Deployment as a **web service or email plugin**  
- Model optimization for edge devices  

## License
This project is open-source and available for educational purposes.

## Acknowledgments
- TensorFlow and Keras teams  
- Public spam datasets (e.g., Enron)  
- NLP libraries: NLTK, SpaCy  
