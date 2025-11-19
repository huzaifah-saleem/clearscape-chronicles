# Arabic Sentiment Analysis with AraBERT Tokenizer and CatBoost

## 📖 Overview

This project demonstrates **Arabic sentiment analysis on Twitter data** using a hybrid approach that combines:
- **AraBERT** pre-trained transformer model for text tokenization and embedding generation
- **CatBoost** gradient boosting classifier for sentiment classification
- **Teradata Vantage BYOM (Bring Your Own Model)** for in-database model scoring

The solution leverages Teradata's **ClearScape Analytics** to perform end-to-end sentiment analysis directly within the database, eliminating data movement and enabling scalable, production-ready ML workflows. The model achieves **~89% accuracy** on Arabic tweet sentiment classification.

### Key Features:
✅ Arabic text preprocessing with language-specific stopword removal  
✅ AraBERT embeddings (768-dimensional vectors) for contextual understanding  
✅ CatBoost classifier trained on embeddings  
✅ ONNX model export for Teradata BYOM deployment  
✅ In-database scoring with Teradata's ONNXPredict function  

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
- Understanding of NLP and sentiment analysis concepts
- Familiarity with transformer models (optional but helpful)

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
pip install pandas numpy torch nltk transformers onnxruntime tqdm teradataml catboost scikit-learn matplotlib seaborn
```

**Package Breakdown:**
- `pandas`, `numpy` - Data manipulation and numerical operations
- `torch` - PyTorch framework for deep learning
- `nltk` - Natural language processing toolkit (Arabic stopwords)
- `transformers` - Hugging Face library for AraBERT model
- `onnxruntime` - ONNX model runtime for inference
- `tqdm` - Progress bar for long-running operations
- `teradataml` - Teradata Python library for database operations
- `catboost` - Gradient boosting classifier
- `scikit-learn` - Machine learning utilities and metrics
- `matplotlib`, `seaborn` - Data visualization libraries

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

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `tid` | Integer | Tweet ID (unique identifier) |
| `docstring` | Text | Raw Arabic tweet text |
| `Sentiment` | Integer | Binary sentiment label (0=Negative, 1=Positive) |

### Preprocessing & Preparation Steps Involved

1. **Text Cleaning**: Remove URLs, mentions, hashtags, emojis, and special characters
2. **Stopword Removal**: Filter out common Arabic stopwords using NLTK
3. **Normalization**: Standardize Arabic text (diacritics, character forms)
4. **Tokenization**: Use AraBERT tokenizer for subword tokenization
5. **Embedding Generation**: Convert tokens to 768-dimensional embeddings

---

## 🔄 Visual Workflow

<div align="center">
  <img width="897" height="380" alt="image" src="https://github.com/user-attachments/assets/de6ddc41-e96a-4361-82ee-c3ada90a37b6" />
</div>

---

## 🚀 Teradata Vantage Technology: BYOM (Bring Your Own Model)

### What is BYOM?

**Bring Your Own Model (BYOM)** is a ClearScape Analytics capability that enables data scientists to:
- Deploy externally trained ML models (PyTorch, TensorFlow, scikit-learn, etc.) directly into Teradata Vantage
- Score data at scale using **in-database execution** without data movement
- Leverage Teradata's massively parallel processing (MPP) architecture for fast inference
- Integrate custom models with SQL workflows seamlessly

### BYOM in This Project

This notebook demonstrates BYOM using the **ONNX (Open Neural Network Exchange)** format:
```

#### 3. **In-Database Scoring with ONNXPredict**
```python
# Score embeddings using stored ONNX model
predictions = tdml.ONNXPredict(
    data = tdml.DataFrame("arabic_sentiment_data"),
    modeldata = tdml.DataFrame("arabic_sentiment_classifier"),
    accumulate = ["tid", "Sentiment"],
    model_input_fields_map = "features=emb_0:emb_767"  # 768-dim embeddings
)
```

### Benefits of BYOM for This Use Case

✅ **No Data Movement**: Process millions of Arabic tweets without extracting data from Teradata  
✅ **Parallel Execution**: Score across all AMPs simultaneously for maximum throughput  
✅ **Production Ready**: Models are versioned, stored, and governed within the database  
✅ **Real-Time Scoring**: Integrate with operational applications via SQL queries  
✅ **Model Monitoring**: Track performance metrics and drift detection in-database  

### Supported Formats
- **ONNX** (used in this project)
- **H2O MOJO**
- **PMML**
- **Custom UDFs**

---

## 📚 Documentation

### Official Guide on BYOM 

- **[BYOM Documentation](https://docs.teradata.com/r/Enterprise_IntelliFlex_Lake_VMware/Teradata-VantageTM-Bring-Your-Own-Model-User-Guide/Welcome-to-Bring-Your-Own-Model))**

### Model & Framework Resources

- **[CatBoost Documentation](https://catboost.ai/docs/)**

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
   jupyter lab Sentiment_analysis_ARABERT_TokenizerOnly.ipynb
```

2. **Configure Teradata connection and run the code**
```python
   import teradataml as tdml
   
   tdml.create_context(
       host='your-host.clearscape.teradata.com',
       username='your-username',
       password='your-password'
   )
```

## 📊 Results

### Model Performance

**CatBoost with AraBERT Embeddings (In-Database Scoring)**
- **Accuracy**: 88.87%
- **F1 Score**: 89.99%
- **Inference Speed**: Scalable across Teradata MPP architecture

### Key Insights

✅ AraBERT embeddings capture Arabic semantic context effectively  
✅ CatBoost handles imbalanced sentiment distribution well  
✅ BYOM enables production deployment without performance degradation  
✅ In-database scoring eliminates ETL overhead and latency  

---

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
