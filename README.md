# Mood Detector with LSTM

A deep learning project to classify text sentiment (e.g., from tweets) as **Positive** or **Negative** using a Long Short-Term Memory (LSTM) neural network.  
This project demonstrates the basics of Natural Language Processing (NLP), sentiment analysis, and recurrent neural network modeling with TensorFlow and Keras.

---

## Table of Contents
- [Introduction / Motivation](#introduction--motivation)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Introduction / Motivation
Understanding sentiment in text is a fundamental NLP task with applications in customer feedback analysis, social media monitoring, and more.  
This project automates the process of classifying text into emotional categories using a deep learning approach.

**Problem it solves:**
- Automates sentiment classification for large text datasets.
- Provides a hands-on example of an LSTM model applied to a real-world NLP task.

**Key highlights:**
- End-to-end NLP workflow: data loading, cleaning, tokenization, and training.
- Implementation of an LSTM-based Keras Sequential model.
- Text preprocessing pipeline tailored for sentiment analysis.
- Visualization of model training history (accuracy and loss).

---

## Features
- **Text Preprocessing:** Removes URLs, mentions, stopwords; tokenizes text.
- **Word Embeddings:** Embedding layer to represent words as dense vectors.
- **Sequence Modeling:** LSTM layer captures contextual dependencies in text.
- **Model Training & Evaluation:** Uses accuracy and loss metrics.
- **Modular Code:** Notebook structured into clear, logical sections.

---

## Dataset
Requires a CSV or Excel file with at least two columns:
1. **Text** — raw input sentence.
2. **Sentiment Label** — e.g., "Positive" or "Negative".

The notebook loads this data into a pandas DataFrame and applies cleaning and preprocessing.

---

## Project Structure
```
Mood_Detector_LSTM.ipynb   # Main Jupyter Notebook
README.md                  # This file
```
Optional directories to add:
```
data/     # For dataset storage
src/      # Python scripts (if modularizing code)
models/   # Trained model weights
```

---

## Installation
**Prerequisites:**
- Python 3.8+
- pip or conda
- Jupyter Notebook or JupyterLab

**Setup:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -U pip
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn nltk
```

**Download NLTK data:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

**Run notebook:**
```bash
jupyter notebook
```

---

## Usage
1. Place your dataset in the project directory.
2. Open `Mood_Detector_LSTM.ipynb`.
3. Update the file path in the data loading cell.
4. Run all cells in order:
   - Import libraries and load data
   - Preprocess text
   - Tokenize & pad sequences
   - Build, compile, and train the model
   - View accuracy/loss plots

---

## Model Architecture
- **Embedding Layer** — integer-encoded vocabulary to dense vectors.
- **LSTM Layer** — captures sequence dependencies.
- **Dropout Layer** — reduces overfitting.
- **Dense Layer** — fully connected processing.
- **Output Dense Layer** — sigmoid activation for binary classification.

---

## Results
Trained for 5 epochs.  
Validation accuracy: ~50%, indicating the model is not learning effectively yet.

**Possible Improvements:**
- Hyperparameter tuning (embedding size, LSTM units, learning rate)
- Pre-trained embeddings (GloVe, Word2Vec)
- Bidirectional or stacked LSTMs
- Longer training with early stopping
- Data augmentation

---

## Contributing
Contributions are welcome:
1. Fork the repository.
2. Create a feature branch.
3. Commit changes.
4. Push to your branch.
5. Open a Pull Request.

---

## License
This project is recommended to be licensed under the MIT License.

---

## Acknowledgements
This project was completed as part of the Data Science and AI training program at **Internselite**.  
Special thanks to my mentor **Ayush Srivastava** for invaluable guidance and support.  
Libraries used: TensorFlow, Keras, Scikit-learn, Pandas, NLTK, Matplotlib.

---

## Contact
**Author:** Arindam Deka  
**Email:** arindamd993@gmail.com  
**GitHub:** [ArindamDeka09](https://github.com/ArindamDeka09)  
For questions or issues, please open an issue in this repository.
