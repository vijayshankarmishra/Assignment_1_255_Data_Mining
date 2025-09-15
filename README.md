# Data-Mining-Assignment-1

Kaggle Dataset : https://www.kaggle.com/datasets/tarekmasryo/youtube-shorts-and-tiktok-trends-2025 \
ChatGPT Chat : https://chatgpt.com/share/68c74ffb-da24-8006-a28f-b1dd8e5f9312 \
Medium Blog : https://medium.com/@vijayshankar.mishra/predicting-the-popularity-of-short-videos-with-data-science-43cb44c1a2d4  

A comprehensive data science project analyzing the factors influencing short video popularity across YouTube Shorts and TikTok, exploring engagement patterns, and building ML models to predict video view counts.  

---

## 📊 Visual Insights

### Distribution of Video Views
![Distribution Views](Reports/figures/Distribution_Views.png)

### Distribution of log1p(Views)
![Distribution of Views](Reports/figures/Distribution_Log1p(Views).png)

### Likes 
![Likes vs Views](Reports/figures/Distribution_Likes.png)

### Comments
![Shares vs Views](Reports/figures/Distribution_comments.png)

### Shares
![Comments vs Views](Reports/figures/Distribution_Shares.png)

### CreatorFollowers
![Feature Importance](Reports/figures/Distribution_CreatorFollowers.png)

---

## 🗂 Dataset Overview

| Feature | Description |
|---------|-------------|
| **Views** | Target variable – number of views a video received |
| **Likes** | Number of likes the video gained |
| **Comments** | Number of comments on the video |
| **Shares** | Number of shares/reposts |
| **Duration (seconds)** | Length of the video |
| **Platform** | YouTube Shorts or TikTok |
| **Country / Region / Language** | Demographic info |
| **Hashtags / Title / Description** | Textual metadata |

---

## 🧠 Approach & Techniques

### 1. Business Understanding

Predict **video views** based on engagement and metadata features.  
Helps creators and marketers understand what drives video popularity and design better content strategies.  

### 2. Data Preparation

- Dropped irrelevant identifiers (video IDs).  
- Scaled numerical features like likes, comments, shares, and duration.  
- Encoded categorical features such as platform and country.  
- Cleaned text-based fields (hashtags, titles) for potential feature engineering.  

### 3. Feature Engineering

- **Engagement Ratios** (likes/views, comments/views) to normalize influence.  
- **Hashtag Encoding** to represent presence of trending tags.  
- **Duration Buckets** (short, medium, long) to capture non-linear length effects.  

### 4. Model Selection

- **Linear Regression** as a baseline.  
- **Random Forest Regressor** for capturing non-linear relationships.  
- **XGBoost Regressor** for higher accuracy and feature importance insights.  

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

```

### 5. Evaluation

- **Metrics**: RMSE and R² Score  
- Random Forest and XGBoost outperformed the baseline.  
- Feature importance showed **likes, shares, and comments** as the top drivers of video views.  

---

## 📈 Why These Features?

- **Likes, Comments, Shares** → Strong engagement proxies.  
- **Duration** → Medium-length videos often perform best.  
- **Platform** → TikTok vs YouTube Shorts reward different patterns.  
- **Hashtags / Metadata** → Indicate trends and discoverability.  

---

## 🚀 Future Improvements

- Use **NLP embeddings** for titles, descriptions, and hashtags.  
- Perform **hyperparameter tuning** on Random Forest / XGBoost.  
- Build a **Streamlit dashboard** for creators to enter video details and get predicted views.  
- Extend dataset with **posting time** and **trending data** for more accurate predictions.
