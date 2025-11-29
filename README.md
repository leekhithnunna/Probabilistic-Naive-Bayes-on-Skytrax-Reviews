# Probabilistic-Naive-Bayes-on-Skytrax-Reviews
This project builds a full-on Probabilistic Naïve Bayes Sentiment Classification System for real-world airline industry reviews taken from Skytrax. We work with four datasets — Airline, Airport, Lounge, and Seat — each containing numeric ratings, text reviews, and categorical info from thousands of passengers.

# ⭐ Probabilistic Naïve Bayes on Skytrax Reviews

## 🚀 Project Overview
This project builds a **multi-class sentiment classifier** for four Skytrax review datasets:

- ✈️ **Airline**
- 🛫 **Airport**
- 🛋️ **Lounge**
- 💺 **Seat**

Each dataset contains:
- Numeric rating features (comfort, staff service, cleanliness…)
- Text reviews
- Categorical / binary metadata

Our models classify each review into:
- **0 → Negative**
- **1 → Neutral**
- **2 → Positive**

The ML pipeline uses multiple Naïve Bayes flavours:
- **GaussianNB** → numeric rating features  
- **MultinomialNB** → text (Word2Vec + Bag-of-Centroids)  
- **BernoulliNB** → binary/categorical  
- **Hybrid Model** → weighted fusion of GNB + MNB probabilities  

---

## 🔧 Step 1 — Target Construction
Each dataset includes an **overall_rating (1–10)**.  
We convert this into a 3-class target:

1–4 → 0 (Negative)
5–7 → 1 (Neutral)
8–10 → 2 (Positive)


Invalid or missing rating rows are removed.

Scripts used:
- `airport_with_target.ipynb`
- `airline_with_target.ipynb`
- `lounge_with_target.ipynb`
- `seat_with_target.ipynb`

---

## 🧼 Step 2 — Preprocessing & Feature Engineering

### 🔹 A) Numeric Features → GaussianNB
- Select all `*_rating` columns  
- Drop features with **>50% missing values**  
- Median-impute missing entries (computed from training split only)  
- Apply an **80–20 stratified split**  

### 🔹 B) Text Features → MultinomialNB
- Extract the text review (`content`)  
- Clean → tokenize → prepare training text  
- Train **Word2Vec** on training reviews  
- Cluster embeddings using **MiniBatch KMeans**  
- Convert reviews into **Bag-of-Centroids (BoC)** vectors  
- Save BoC matrices as NumPy files  

### 🔹 C) Categorical Features → BernoulliNB
- One-hot encode: traveller type, cabin flown, seat layout, etc.  
- Convert `recommended` into binary 0/1  
- Build a sparse indicator matrix  

---

## 🤖 Step 3 — Model Training

### **1️⃣ Gaussian Naïve Bayes**
Input: numeric CSV  
Outputs:
- Accuracy  
- Macro/Weighted F1  
- Log-loss  
- Confusion matrix  

### **2️⃣ Multinomial Naïve Bayes**
Inputs:  
- `X_train_boc.npy`  
- `y_train.npy`  

Outputs:
- Class probabilities  
- Performance metrics  
- Normalized confusion matrix  

### **3️⃣ Bernoulli Naïve Bayes**
- Best for binary/categorical-only datasets (e.g., Lounge)  

### **4️⃣ Hybrid Model**
Weighted fusion:
P_hybrid = α * P_GNB + (1 - α) * P_MNB

Prediction = class with highest probability.

---

## 📊 Step 4 — Results Summary (Examples)

| Dataset | Best Model | Accuracy | Macro F1 | Comment |
|--------|------------|----------|----------|---------|
| Airline | Hybrid | ~0.75 | ~0.57 | Text + numeric helps |
| Airport | Hybrid | ~0.76 | ~0.60 | Balanced feature set |
| Lounge | GaussianNB | ~0.87 | ~0.62 | Numeric ratings are very clean |
| Seat | Hybrid | ~0.81 | ~0.77 | Text boosts performance |

---

🙌 Authors
- Leekhith Nunna
