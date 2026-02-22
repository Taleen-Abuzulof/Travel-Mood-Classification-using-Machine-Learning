# 🌍 Travel Mood Classification Using Machine Learning  
### A Comparative Study of Classical NLP Models with Imbalance Handling

**Authors:**  
Taleen Abuzoluf  
Mayar Jafar  

---

## 📖 Abstract

This project investigates the problem of predicting emotional states (**Mood/Emotion**) from travel descriptions using supervised machine learning techniques. The study explores classical NLP pipelines including TF-IDF representations, sentence embeddings, and multiple classification algorithms.

Special emphasis is placed on:

- Robust dataset assembly and cleaning  
- Text preprocessing and normalization  
- Class imbalance mitigation using SMOTE  
- Hyperparameter optimization  
- Comparative evaluation across models  

This work demonstrates a complete end-to-end ML pipeline from raw CSV files to evaluated classification models.

---

## 🎯 Problem Statement

Given a structured dataset containing travel-related metadata and textual descriptions, the goal is to:

> **Predict the corresponding Mood/Emotion category based solely on the textual description.**

This is formulated as a multi-class supervised classification problem.

## 📂 Repository Structure

```
├── Final_ML_3.ipynb
├── combined_clean.csv
├── README.md
└── requirements.txt
```
---

## 📊 Dataset Construction

### Final Dataset Columns

 **Image URL**  
  A direct link to the image associated with the travel description. Used as a reference to the visual context of the entry.

 **Description**  
  A textual narrative describing the travel experience. This is the primary input feature used for predicting the emotional category.

 **Country**  
  The country where the travel experience took place. Provides geographical context.

 **Weather**  
  The weather condition during the experience (e.g., sunny, rainy, snowy). May influence the emotional tone.

 **Time of Day**  
  Indicates when the experience occurred (e.g., morning, afternoon, evening, night).

 **Season**  
  The season during which the experience happened (e.g., summer, winter, spring, autumn).

 **Activity**  
  The main activity performed during the experience (e.g., hiking, sightseeing, relaxing).

 **Mood/Emotion (Target Variable)**  
  The labeled emotional category associated with the description. This is the variable the model aims to predict.
  
### Data Cleaning Strategy

Before any modeling could begin, the dataset needed careful preparation. The raw files came from multiple sources and were not perfectly consistent. Some files were saved in different text encodings, so the first step was teaching the system how to read them properly. We allowed it to intelligently decode each file using common formats such as `utf-8`, `cp1252`, and `latin1`, ensuring no valuable information was lost due to encoding mismatches.

As we explored the data further, we discovered rows that were incomplete, corrupted, or improperly formatted. These problematic entries were carefully filtered out to maintain dataset integrity. Clean data is the foundation of reliable machine learning, so this step was essential.

Next, we addressed duplication. Since the dataset was assembled from multiple folders, some entries appeared more than once. Duplicate rows were identified and removed to avoid biasing the model or inflating performance metrics. We also enforced mandatory fields. Each entry had to contain at least a valid `Image URL` and a `Description`. Without these two core components, the record would not meaningfully contribute to the learning task. Any entry missing these fields was excluded.

Finally, categorical columns such as country, weather, activity, and season were standardized. Variations in spelling, capitalization, or formatting were corrected to ensure consistency. For example, entries like “Summer” and “summer” were unified, preventing artificial category fragmentation.

By the end of this process, the dataset transformed from a collection of messy CSV files into a clean, structured, and reliable foundation ready for analysis and modeling. The final cleaned dataset was exported as combined_clean.csv


---

## 🔎 Exploratory Data Analysis (EDA)

In EDA, we focused on analysing Class distribution and Detecting imbalance across mood categories using Visualization techniques like count plots.  
The analysis revealed noticeable class imbalance in the target variable, motivating the use of oversampling techniques.

---

## 🧹 Text Preprocessing Pipeline

The textual descriptions were processed using:

- Lowercasing  
- Regex cleaning  
- Stopword removal (NLTK)  
- Lemmatization  
- Token normalization  

The target variable was label-encoded for modeling.

---

## 🧠 Feature Extraction and Engineering

Feature engineering played a central role in this project, as the performance of text classification models is highly dependent on how textual information is represented numerically. Since machine learning algorithms cannot directly process raw text, the descriptions had to be transformed into meaningful vector representations.

To better understand the impact of representation choice, two fundamentally different approaches were evaluated: a traditional statistical method (TF-IDF) and a modern deep-learning-based semantic embedding approach (Sentence Transformers).

---

### 1️⃣ TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) is a classical and widely used technique in Natural Language Processing. It converts text into a high-dimensional sparse vector where each dimension corresponds to a unique word in the vocabulary.

The core idea behind TF-IDF is:

- Words that appear frequently in a document are important.
- Words that appear in many documents are less discriminative.
- Rare but meaningful words receive higher importance scores.

This representation captures term importance at the lexical level, making it effective for many traditional machine learning classifiers. Because TF-IDF produces sparse vectors, cosine similarity was used as the distance metric, particularly for KNN classification.

Advantages:
- Simple and interpretable
- Computationally efficient
- Strong baseline for text classification

Limitations:
- Does not capture word order
- Cannot understand semantic similarity (e.g., "happy" vs "joyful") or context.

---

### 2️⃣ Sentence Embeddings (Sentence Transformers)

To move beyond surface-level word statistics, we also explored dense semantic representations using pre-trained Sentence Transformer models.

Unlike TF-IDF, sentence embeddings encode entire descriptions into dense, low-dimensional vectors that capture contextual and semantic meaning. These models are trained on large-scale corpora using transformer architectures, allowing them to understand relationships between words beyond simple frequency counts.

For example, descriptions containing words like:
- "peaceful", "calm", "relaxing"

may be positioned closely in embedding space, even if they do not share identical vocabulary.

Advantages:
- Captures contextual meaning
- Encodes semantic similarity
- More robust to vocabulary variations

Limitations:
- Higher computational cost
- Less interpretable than TF-IDF

---

### 🔎 Why Compare Both?

By evaluating both TF-IDF and Sentence Embeddings, this project explores the trade-off between classical statistical NLP methods and modern transformer-based representations. This comparison helps determine whether semantic richness translates into measurable improvements in classification performance for this specific mood prediction task. as discovered later, the TF-IDF produced better results mainly because the dataset descriptions had no deep semantic meaning and can be predicted by some presence individual words.

---

## ⚖️ Class Imbalance Handling

To address class imbalance:

- **SMOTE (Synthetic Minority Oversampling Technique)** was applied and Implemented using `imblearn` pipelines, Oversampling was performed only on training data. This improved macro-level evaluation metrics.
- Choosing the `class = 'balanced'` option in some ML archetecture was also implemented. 

---

## 🤖 Models Evaluated

The following classifiers were implemented:

- **K-Nearest Neighbors (KNN)**  
  - Cosine distance metric  
  - Distance-weighted voting  

- **Logistic Regression**

- **Support Vector Machine (SVM)**  

Hyperparameters were optimized using:

- `GridSearchCV`
- Cross-validation

---

## 📈 Evaluation Metrics

Models were evaluated using Accuracy, F1-Score (Macro & Weighted), Classification Report, and Confusion Matrix.
due to the class imbalance, the F1-score Macro and F1-score for minority classes were choosen as the main evaluation metrics.


---


## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn  
- Sentence-Transformers  
- NLTK  
- Matplotlib  
- Seaborn  

---

## ▶️ Reproducibility

### 1) Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2) Install the dependancies
```bash
pip install -r requirements.txt
```
### 3) Run the notebook
```bash
jupyter notebook Final_ML_3.ipynb
```




