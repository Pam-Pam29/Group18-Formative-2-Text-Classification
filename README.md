# Text Classification with Different Architectures and Multiple Word Embeddings

**Comparative Analysis of Different Architectures Across Different Word Embedding Techniques**


##  Project Overview

This project implements and evaluates **Different Architectures** for text classification using multiple word embedding techniques. The goal is to conduct a comparative analysis to understand how different embedding methods impact model performance on sentiment classification tasks.

### Research Questions
1. How do different word embedding techniques (TF-IDF, Skip-gram, CBOW, GloVe) affect BiRNN performance?
2. Which embedding method provides the best semantic representation for sentiment analysis?
3. What are the computational trade-offs between different embedding approaches?

---

##  Team Members & Contributions

| Team Member | Model Assignment | Embeddings Tested | Key Contributions |
|-------------|-----------------|-------------------|-------------------|
| **Victoria Fakunle** | BiRNN (Bidirectional RNN) | TF-IDF, Skip-gram, CBOW, GloVe | Model implementation, hyperparameter tuning, results analysis |
| **MUGISHA Samuel** | LSTM | TF-IDF, Skip-gram, CBOW | [Contributions] |
| **NIWEMWANA Aline Innocente** | Traditional Model | TF-IDF, Skip-gram, GloVe | [Contributions] |
| **GATWAZA Jean Robert** | GRU | TF-IDF, CBOW, GloVe | [Contributions] |



**Contribution Tracker:** [https://docs.google.com/spreadsheets/d/1y8lgGQ-klK1ySLxJcrCgK7LVt5GNOl-T_QqozzQhwdI/edit?gid=0#gid=0]

---

##  Dataset

**Dataset Name:** Financial Sentiment Analysis

**Source:**  Kaggle

**Description:** 
- **Task:** Text Classification (3 classes: Negative, Neutral, Positive)


**Preprocessing Steps:**
1. Text cleaning (lowercasing, removing special characters, URLs)
2. Tokenization
3. Stopword removal
4. Lemmatization/Stemming
5. Sequence padding/truncation to max_length=100

---


##  Installation & Requirements

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository:**

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download GloVe embeddings:**
```bash
# Download GloVe 6B 300d
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/embeddings/
```

### Dependencies
```
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
gensim==4.3.2
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
```

---

##  Word Embeddings Explored

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- **Type:** Traditional sparse representation

### 2. Skip-gram (Word2Vec)
- **Type:** Predictive embedding (context → word)

### 3. CBOW (Continuous Bag of Words)
- **Type:** Predictive embedding (word → context)
- **Vocabulary:** Custom-trained on dataset

### 4. GloVe (Global Vectors)
- **Type:** Count-based pre-trained embedding
- **Source:** Stanford GloVe 6B
- **Pre-trained on:** Wikipedia 2014 + Gigaword 5
- **Coverage:** ~400k words

---


### Evaluation Metrics
- **Accuracy:** Overall classification accuracy
- **Precision:** Class-wise precision scores
- **Recall:** Class-wise recall scores
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visualization of classification errors

---


##  References

### Word Embeddings
1. Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space." *arXiv preprint arXiv:1301.3781*.
2. Pennington, J., Socher, R., & Manning, C. (2014). "GloVe: Global Vectors for Word Representation." *EMNLP*.
3. Ramos, J. (2003). "Using TF-IDF to Determine Word Relevance in Document Queries."

### Neural Network Architectures
4. Schuster, M., & Paliwal, K. K. (1997). "Bidirectional Recurrent Neural Networks." *IEEE Transactions on Signal Processing*.
5. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*.
6. Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder." *EMNLP*.

### Text Classification
7. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification." *EMNLP*.
8. Zhang, X., Zhao, J., & LeCun, Y. (2015). "Character-level Convolutional Networks for Text Classification." *NIPS*.

### Additional Resources
9. [Keras Documentation](https://keras.io/)
10. [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
11. [Stanford NLP Group - GloVe](https://nlp.stanford.edu/projects/glove/)

---


##  License

This project is for academic purposes only. Dataset and code usage subject to original source licenses.

---

##  Acknowledgments

- Course Instructor: Kevin

- Stanford NLP Group for GloVe embeddings
- Kaggle for providing the dataset

---

##  Citation

If you use this work, please cite:

```bibtex
@misc{birnn_text_classification_2025,
  title={Comparative Analysis of BiRNN with Multiple Word Embeddings for Text Classification},
  author={[Your Names]},
  year={2025},
  institution={[Your University]},
  course={[Course Code]}
}
```

---


**Course:** [Machine Learning Techniques 1]

**Semester:** [January 2026]
