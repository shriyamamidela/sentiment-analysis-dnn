
# Sentiment Analysis using Deep Neural Networks

This project explores and compares the performance of four different deep learning architectures—**CNN**, **LSTM**, **MLP**, and **Autoencoder**—for binary sentiment classification of tweets (positive or negative).


## 📌 Objective
To classify tweet sentiments (positive or negative) using deep learning models and compare their performance across different architectures and hyperparameter settings.

## 📊 Dataset
- **Source:** [Twitter Sentiment Analysis Dataset – Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv)
- **Filename:** `train.csv`
- **Format:** CSV
- **Encoding:** ISO-8859-1
- **Classes:** `positive`, `negative` (neutral excluded)
- **Used Features:** `text`, `sentiment`
- **Dropped/Unused Columns:** `time of tweet`, `age of user`, `country`, `population`, `land area`, `density`


## 🛠️ Technologies Used
- Python, PyTorch, torchtext
- pandas, scikit-learn, NumPy

## 🧹 Preprocessing Steps
- Removed URLs, special characters, digits
- Lowercased text
- Tokenized using `torchtext`
- Built vocabulary with `<unk>` for rare words
- Encoded sentiment: `negative → 0`, `positive → 1`
- Padded tweets to a fixed token length

## 🧠 Models Implemented
### 1. CNN (Convolutional Neural Network)
- Used Conv1D layers with varying kernel sizes
- Tuned hyperparameters: `embed_dim`, `num_filters`, `kernel_size`, `learning_rate`
- Best Accuracy: **92.19%** (train), **84.97%** (test)

### 2. LSTM (Long Short-Term Memory)
- Bidirectional LSTM with 1–2 layers
- Hyperparameter tuning done for `embed_dim`, `hidden_dim`, `num_layers`, `learning_rate`
- Best Accuracy: **86.25%**

### 3. MLP (Multi-Layer Perceptron)
- Used averaged embeddings with 2 dense layers
- Best Accuracy: **85.36%**

### 4. Autoencoder-Based Classifier
- Encoder compresses to bottleneck → classifier head
- Decoder exists but unused in classification
- Best Accuracy: **87.28%**

## 📈 Result Summary
| Model        | Best Test Accuracy |
|--------------|--------------------|
| CNN          | 84.97%             |
| LSTM         | 86.25%             |
| MLP          | 85.36%             |
| Autoencoder  | 87.28%             |

## 📚 Report
Detailed report is available in `DNN_report.docx`.

---

<pre> ## 📦 File Structure ``` sentiment-analysis-dnn/ ├── Sentiment_Models.ipynb # Jupyter Notebook with all model code ├── DNN_report.docx # Detailed project report ├── train.csv # Twitter sentiment dataset └── README.md # Project overview and documentation ``` </pre>


---

## 🏃‍♀️ How to Run
1. Clone the repo  
   `git clone https://github.com/shriyamamidela/sentiment-analysis-dnn.git`

2. Install dependencies  
   `pip install torch pandas scikit-learn`

3. Run the notebook  
   Open `Sentiment_Models.ipynb` and run all cells.

---

## 💡 Future Work
- Integrate pre-trained embeddings (like GloVe)
- Try attention-based models (e.g., Transformers)
- Deploy as a web application

---

