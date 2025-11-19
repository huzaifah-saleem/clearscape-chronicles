# Arabic Sentiment Analysis with Stacked Meta-Learner Ensemble

## 📖 Overview

This project demonstrates an **advanced Arabic sentiment analysis on Twitter data** using a **stacked ensemble (meta-learning) approach** that combines multiple machine learning models:

### Ensemble Architecture:
1. **Model 1**: AraBERT embeddings (768-dim) + CatBoost classifier
2. **Model 2**: CountVectorizer + Logistic Regression classifier  
3. **Meta-Learner**: Logistic Regression that learns to combine predictions from Models 1 & 2

The solution leverages **Teradata Vantage BYOM (Bring Your Own Model)** to deploy all three models for in-database scoring, achieving **90.60% accuracy** and **91.53% F1 score** on Arabic tweet sentiment classification.

### Key Features:
✅ **Stacked ensemble** approach combining deep learning embeddings and traditional NLP features  
✅ **AraBERT contextual embeddings** for semantic understanding  
✅ **CountVectorizer** for capturing lexical patterns  
✅ **Meta-learner** that optimally weights base model predictions  
✅ **5-fold cross-validation** for robust model training  
✅ **ONNX export** for all three models (CatBoost + Logistic Regression + Meta-Learner)  
✅ **In-database scoring** with Teradata's ONNXPredict function  

---

## ✅ Prerequisites

### 1️⃣ **Teradata Environment Access**

You need access to a Teradata Vantage environment. Get started with the **ClearScape Analytics Experience**:

🔗 **[Sign up for ClearScape Analytics Experience](https://clearscape.teradata.com/sign-in)**

**Setup Steps:**
1. Visit the link above and create a free account
2. Provision a new environment after logging in
3. During setup, configure your credentials:
   - **Host**: Your Teradata instance URL
   - **Username**: Your chosen username  
   - **Password**: Your chosen password
4. Save these credentials - you'll need them to connect

> 💡 **Note**: ClearScape Analytics Experience provides a fully-functional Teradata Vantage environment with sample datasets.

### 2️⃣ **Technical Skills**

- **Python** (3.8+ recommended)
- **SQL** (Teradata SQL dialect)
- Understanding of NLP and ensemble learning concepts
- Familiarity with transformer models and stacking methods

### 3️⃣ **Development Environment**

Choose your preferred development tool:
- **JupyterLab** (available in ClearScape Analytics Experience)
- **VS Code** with Python and Jupyter extensions
- **PyCharm** Professional with Jupyter support
- Any Python IDE with notebook support

---

## 📦 Installation

### Python Dependencies

Install all required packages using pip:

```bash
pip install pandas numpy torch nltk transformers onnxruntime tqdm teradataml catboost scikit-learn matplotlib seaborn skl2onnx
```

**Package Breakdown:**
- `pandas`, `numpy` - Data manipulation and numerical operations
- `torch` - PyTorch framework for deep learning
- `nltk` - Natural language processing toolkit (Arabic stopwords)
- `transformers` - Hugging Face library for AraBERT model
- `onnxruntime` - ONNX model runtime for inference
- `tqdm` - Progress bar for long-running operations
- `teradataml` - Teradata Python library for database operations
- `catboost` - Gradient boosting classifier for Model 1
- `scikit-learn` - Machine learning utilities, Logistic Regression, and CountVectorizer
- `matplotlib`, `seaborn` - Data visualization libraries
- `skl2onnx` - Convert scikit-learn models to ONNX format

### Download NLTK Data

After installation, download the Arabic stopwords corpus:

```python
import nltk
nltk.download('stopwords')
```

---

## 📊 Data

### Dataset Description

**Arabic Sentiment Analysis Dataset - SS2030**

- **Source**: Arabic tweets dataset (CSV format)
- **Size**: 4,252 tweets
- **Language**: Arabic
- **Sentiment Distribution**:
  - Positive (1): 2,436 tweets (57.3%)
  - Negative (0): 1,815 tweets (42.7%)
- **Train/Test Split**: 70% training (2,975 tweets) / 30% testing (1,276 tweets)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `tid` | Integer | Tweet ID (unique identifier) |
| `docstring` | Text | Raw Arabic tweet text |
| `Sentiment` | Integer | Binary sentiment label (0=Negative, 1=Positive) |

### Preprocessing & Feature Engineering Steps

#### Text Preprocessing (Common for Both Models):
1. **Text Cleaning**: Remove URLs, mentions, hashtags, emojis, and special characters
2. **Arabic Character Filtering**: Keep only Arabic Unicode characters (U+0600 to U+06FF)
3. **Stopword Removal**: Filter out common Arabic stopwords using NLTK
4. **Normalization**: Standardize whitespace and character forms

#### Model 1 - AraBERT Features:
5. **Tokenization**: Use AraBERT tokenizer for subword tokenization
6. **Embedding Generation**: Extract [CLS] token embeddings (768 dimensions)
7. **Feature Storage**: Store embeddings in Teradata for training

#### Model 2 - CountVectorizer Features:
5. **Vectorization**: Apply CountVectorizer to generate term frequency features
6. **Feature Selection**: Use SelectKBest with chi-squared test (top 500 features)
7. **Normalization**: Apply StandardScaler to normalize feature values

---

## 🔄 Visual Workflow

<div align="center">
   
<img width="897" height="380" alt="image" src="https://github.com/user-attachments/assets/e4b4fab4-a653-4201-ba20-f5ad6d6d3a8b" />

</div>

---

## 🚀 Teradata Vantage Technology: BYOM (Bring Your Own Model)

### What is BYOM?

**Bring Your Own Model (BYOM)** is a ClearScape Analytics capability that enables data scientists to:
- Deploy externally trained ML models (PyTorch, TensorFlow, scikit-learn, etc.) directly into Teradata Vantage
- Score data at scale using **in-database execution** without data movement
- Leverage Teradata's massively parallel processing (MPP) architecture for fast inference
- Integrate custom models with SQL workflows seamlessly

### Benefits of BYOM for This Ensemble Approach

✅ **No Data Movement**: Process millions of Arabic tweets without extracting data from Teradata  
✅ **Parallel Execution**: Score across all AMPs simultaneously for maximum throughput  
✅ **Multi-Model Orchestration**: Deploy and manage multiple models as a unified pipeline  
✅ **Production Ready**: All models are versioned, stored, and governed within the database  
✅ **Real-Time Ensemble Scoring**: Integrate stacked predictions with operational applications via SQL  
✅ **Model Monitoring**: Track performance metrics and drift detection for each model in-database  

### Ensemble Learning Advantages

The meta-learner approach provides several benefits over single models:

- **Diversity**: Combines deep learning (AraBERT) with traditional NLP (CountVectorizer)
- **Robustness**: Meta-learner learns optimal weighting of base model predictions
- **Performance**: Achieves 90.60% accuracy vs. 88.87% from AraBERT alone
- **Interpretability**: Can analyze which model contributes most to final predictions
- **Generalization**: Reduces overfitting through model diversity

### Supported Formats
- **ONNX** (used in this project for all three models)
- **H2O MOJO**
- **PMML**
- **Custom UDFs**

---

## 📚 Documentation

### Official Guide on BYOM

- **[BYOM Documentation](https://docs.teradata.com/r/Enterprise_IntelliFlex_Lake_VMware/Teradata-VantageTM-Bring-Your-Own-Model-User-Guide/Welcome-to-Bring-Your-Own-Model)**

### Model & Framework Resources

- **[CatBoost Documentation](https://catboost.ai/docs/)**
- **[scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)**
- **[ONNX Official Website](https://onnx.ai/)**
- **[skl2onnx Documentation](http://onnx.ai/sklearn-onnx/)**

### Community & Support

- **[Teradata Community Forums](https://support.teradata.com/community)**
- **[Teradata Developer Portal](https://developers.teradata.com/)**
- **[Stack Overflow - Teradata Tag](https://stackoverflow.com/questions/tagged/teradata)**

### Related Examples

- [Teradata ML Engine Examples](https://github.com/Teradata)
- [ClearScape Analytics Use Cases](https://www.teradata.com/platform/clearscape-analytics)

---

## 📝 Usage

### Quick Start

1. **Clone the repository and open the notebook**
   ```bash
   jupyter lab Sentiment_Classification_Using_Meta_Learner.ipynb
   ```

2. **Configure Teradata connection**
   ```python
   import teradataml as tdml
   import getpass as gp
   
   tdml.create_context(
       host='your-host.clearscape.teradata.com',
       username='your-username',
       password=gp.getpass(prompt='Password:')
   )
   ```

3. **Load your Arabic sentiment dataset**
   ```python
   df = pd.read_csv('../Datasets/Arabic Sentiment Analysis Dataset - SS2030.csv')
   ```

4. **Run the notebook cells sequentially** to:
   - Preprocess Arabic text
   - Train Model 1 (AraBERT + CatBoost) with 5-fold CV
   - Train Model 2 (CountVectorizer + LR)
   - Generate base model predictions
   - Train Meta-Learner on stacked predictions
   - Export all models to ONNX
   - Deploy to Teradata Vantage
   - Perform in-database ensemble scoring

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | F1 Score | AUC | Notes |
|-------|----------|----------|-----|-------|
| **Model 1: AraBERT + CatBoost** | 88.87% | 89.99% | - | Deep learning embeddings |
| **Model 2: CountVec + LR** | ~86% | ~87% | - | Traditional NLP features |
| **Meta-Learner (Ensemble)** | **90.60%** | **91.53%** | **96.61%** | Stacked combination |

### Final Meta-Learner Test Results

```
Test Accuracy:  90.60%
Test F1-Score:  91.53%
Test AUC:       96.61%

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.90      0.89       564
           1       0.92      0.91      0.92       712

    accuracy                           0.91      1276
```

### Key Insights

✅ **Ensemble outperforms individual models** by ~2% accuracy gain  
✅ **Meta-learner effectively combines** diverse model strengths  
✅ **AraBERT captures semantic context** while CountVectorizer captures lexical patterns  
✅ **High AUC (96.61%)** indicates excellent probability calibration  
✅ **Balanced performance** across positive and negative classes  
✅ **BYOM enables production deployment** without performance degradation  
✅ **In-database scoring eliminates** ETL overhead and latency  

---

## 🧠 Methodology: Stacked Ensemble (Meta-Learning)

### Why Stacking?

**Stacking** (also called **stacked generalization**) is an ensemble learning technique that:
1. Trains multiple diverse base models on the same dataset
2. Uses base model predictions as features for a meta-learner
3. Meta-learner learns the optimal way to combine base predictions
   
### Training Strategy

1. **Split data** into train (70%) and test (30%) sets
2. **Train base models** using 5-fold cross-validation on training data
3. **Generate out-of-fold predictions** to avoid overfitting
4. **Train meta-learner** on base model predictions
5. **Evaluate ensemble** on held-out test set

### Feature Engineering for Meta-Learner

The meta-learner receives 8 input features:
```python
[
    'model1_predicted',              # Binary prediction from AraBERT+CatBoost
    'model1_predicted_probability',  # Confidence score
    'model1_prob_0',                 # Probability of class 0
    'model1_prob_1',                 # Probability of class 1
    'model2_predicted',              # Binary prediction from CountVec+LR
    'model2_predicted_probability',  # Confidence score
    'model2_prob_0',                 # Probability of class 0
    'model2_prob_1'                  # Probability of class 1
]
```

## 📧 Contact

**Author**: Huzaifah - Data Scientist @ Teradata  
**Questions?**: Open an issue in the repository or contact Teradata support

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

<div align="center">
  <p><strong>⭐ If you find this project helpful, please star the repository!</strong></p>
</div>
