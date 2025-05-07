# **🏀 NBA Player Performance Prediction Model**

This project builds a **Linear Regression** model to predict **second-half player performance** using **NBA shot data** from the 2023-24 season. The model leverages first-half statistics, season averages, and opponent strength to estimate second-half scoring, making it valuable for sports analysts, fantasy basketball enthusiasts, and data scientists.

---

## **📁 Project Directory Structure**

```
COMP542_Project/
│
├── Datasets/
│   ├── NBA dataset 23-24 (NEW)/
│   ├── 23-24 NBA dataset.csv
│   ├── 23-24 NBA RANKING.csv
│   └── NBA_2024_per_game(03-01-2024).csv
│
├── linear_performance_predictor.pkl  # Model with multiple features
├── linear_pipeline.pkl               # Serialized pipeline (optional)
├── linear_scaler_1feat.pkl           # Scaler for single-feature models
├── linear_scaler.pkl                 # Scaler for full-feature models
├── NBA Player Performance Project.pdf # Project report (if any)
│
├── NbaNewPredictionsLinear.ipynb    # Jupyter notebook for Linear Regression
├── NbaNewPredictionsXGB.ipynb       # Jupyter notebook for XGBoost
│
├── player_performance_predictor.pkl # Final model for player predictions
├── README.md                        # Project documentation
├── scaler.pkl                       # Final scaler for model
└── train_linear_model.py            # Main training and prediction script
```

---

## **📊 Features Used for Prediction**

### **First-Half Performance:**

- **Total Attempts:** Number of shots taken in the first half
- **Made Shots:** Number of successful shots in the first half
- **Points Scored:** Total points scored in the first half
- **FG%:** Shooting accuracy in the first half
- **3PT Attempt Rate:** Proportion of 3-point shots attempted
- **Average Shot Distance:** Average distance of attempted shots

### **Team Context:**

- **Home/Away Flag:** Indicates if the game was at home
- **Opponent Difficulty:** Inverse of opponent team rank, higher values for tougher teams

### **Season Averages:**

- **Minutes per Game**
- **Points per Game**
- **Assists per Game**
- **Rebounds per Game**
- **Effective Field Goal Percentage (eFG%)**

---

## **⚙️ Requirements**

### **📦 Software Requirements**

- **Python 3.9+**
- **Required Libraries:**

  - **pandas** - Data manipulation
  - **numpy** - Numerical computations
  - **scikit-learn** - Machine learning
  - **joblib** - Model serialization

### **🗄️ Required Datasets**

- **Shot Data:** `23-24 NBA dataset.csv`
- **Team Rankings:** `23-24 NBA RANKING.csv`
- **Season Averages:** `NBA dataset 23-24 (NEW)`

---

## **💻 Hardware Requirements**

This project is not highly resource-intensive, but for best performance, consider:

- **RAM:** 8GB or more
- **CPU:** Recent multi-core processor (e.g., Intel i5, AMD Ryzen 5, or Apple M1)
- **Disk Space:** At least 500MB for project files and datasets

---

## **🚀 Getting Started**

### **Clone the Repository**

```bash
git clone https://github.com/your-repo/NBA-Player-Performance.git
cd NBA-Player-Performance
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Train the Model**

```bash
python train_linear_model.py
```

### **Predict Player Performance**

After training, you can run the prediction function:

```bash
Enter player name to predict: LeBron James
```
