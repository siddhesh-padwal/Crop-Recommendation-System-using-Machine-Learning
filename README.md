# Crop Recommendation System using Machine Learning

## Project Overview

This project develops a machine learning model to recommend suitable crops based on soil and environmental conditions. The model uses soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall to predict the optimal crop from 22 different crop varieties.

The dataset used is `Crop_recommendation.csv`, which contains labeled data for various crops under different conditions.

## Dataset

- **Features (7)**:
  | Feature | Description |
  |-------------|------------------------------|
  | N | Nitrogen content |
  | P | Phosphorus content |
  | K | Potassium content |
  | temperature | Average temperature (°C) |
  | humidity | Relative humidity (%) |
  | ph | Soil pH level |
  | rainfall | Annual rainfall (mm) |

- **Target**: `crop` (22 classes: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee)

## Methodology

1. **Data Loading**: Load `Crop_recommendation.csv` using pandas.
2. **Preprocessing**:
   - Encode crop labels using `LabelEncoder`.
   - Scale features (`N, P, K, temperature, humidity, ph, rainfall`) using `StandardScaler`.
3. **Train-Test Split**: 80-20 split with `random_state=42`.
4. **Model Training**: Random Forest Classifier (`RandomForestClassifier`).
5. **Evaluation**:
   - Classification report.
   - Confusion matrix heatmap.
   - Feature importance plot.

**Script**: `p2.py`

## Results

### Classification Report (Test Set)

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        23
           1       1.00      1.00      1.00        21
           2       1.00      1.00      1.00        20
           3       1.00      1.00      1.00        26
           4       1.00      1.00      1.00        27
           5       1.00      1.00      1.00        17
           6       1.00      1.00      1.00        17
           7       1.00      1.00      1.00        14
           8       0.92      1.00      0.96        23
           9       1.00      1.00      1.00        20
          10       0.92      1.00      0.96        11
          11       1.00      1.00      1.00        21
          12       1.00      1.00      1.00        19
          13       1.00      0.96      0.98        24
          14       1.00      1.00      1.00        19
          15       1.00      1.00      1.00        17
          16       1.00      1.00      1.00        14
          17       1.00      1.00      1.00        23
          18       1.00      1.00      1.00        23
          19       1.00      1.00      1.00        23
          20       1.00      0.89      0.94        19
          21       1.00      1.00      1.00        19
    accuracy                           0.99       440
   macro avg       0.99      0.99      0.99       440
weighted avg       0.99      0.99      0.99       440
```

Mean Absolute Error: 0.06818181818181818
Mean Squared Error: 0.8181818181818182
R2 Score: 0.9795728349655292

- **Accuracy**: 99%
- **mean absolute error**: 0.06818181818181818
- **mean sqaured error**: 0.8181818181818182
- **R2 Score**: 0.9795728349655292
- Excellent performance across most classes, minor misclassifications in a few.

### Feature Importances (Random Forest)

```
Features: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
Importances (%): [10.20, 15.02, 17.69, 7.63, 21.07, 5.39, 23.02]
```

#### How Features Affect Crops

Based on feature importances and dataset patterns:

1. **rainfall (23.02%)** - Most important. High rainfall (>200mm): rice, banana, coconut. Low rainfall (<100mm): chickpea, mothbeans, cotton.
2. **humidity (21.07%)** - High humidity suits rice, coconut (>80%). Low for drier crops like chickpea, mothbeans.
3. **K (17.69%)** - Potassium crucial for fruits/roots: banana, grapes, pomegranate high K.
4. **P (15.02%)** - Phosphorus for growth: legumes (chickpea, pigeonpeas) moderate-high P.
5. **N (10.20%)** - Nitrogen for leaves: rice, maize high N.
6. **temperature (7.63%)** - Tropical: banana, papaya (>25°C); cooler: apple (<25°C).
7. **ph (5.39%)** - Soil acidity: coffee acidic (low pH); others neutral 6-7.

