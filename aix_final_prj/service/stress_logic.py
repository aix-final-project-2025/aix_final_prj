import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings

warnings.filterwarnings('ignore')

# --- 데이터 전처리 함수 (공통 사용) ---
def preprocess_data():
    # Django에서는 manage.py 기준 상대 경로를 사용하는 것이 안전합니다.
    df = pd.read_csv('synthetic_coffee_health_10000.csv')
    df['Health_Issues'] = df['Health_Issues'].fillna('None')
    df1 = df.copy()
    df1 = df1.fillna({'Country': 'Unknown', 'Occupation': 'Unknown', 'Coffee_Intake': 'Unknown'})
    df1 = df1.drop(['ID', 'Caffeine_mg'], axis=1)
    le_country = LabelEncoder()
    le_occupation = LabelEncoder()
    le_coffee = LabelEncoder()
    le_gender = LabelEncoder()
    
    df1['Country'] = le_country.fit_transform(df1['Country'])
    df1['Occupation'] = le_occupation.fit_transform(df1['Occupation'])
    df1['Coffee_Intake'] = le_coffee.fit_transform(df1['Coffee_Intake'])
    df1['Gender'] = le_gender.fit_transform(df1['Gender'])
    
    df1["Stress_Level"] = pd.Categorical(
        df1["Stress_Level"],
        categories=["Low", "Medium", "High"],
        ordered=True
    ).codes
    X = df1[['Age', 'BMI','Country','Coffee_Intake', 'Sleep_Hours', 'Heart_Rate', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption']]
    y = df1['Stress_Level']
    return df1, X, y, le_country, le_occupation, le_coffee 

# --- 1. 분류 모델 분석 로직 ---
def run_classification_analysis():
    df1, X, y, le_country, le_occupation, le_coffee = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(time.time()))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForestClassicfication': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'accuracy': f"{report['accuracy']:.4f}",
            'precision': f"{report['weighted avg']['precision']:.4f}",
            'recall': f"{report['weighted avg']['recall']:.4f}",
            'f1_score': f"{report['weighted avg']['f1-score']:.4f}"
        }
    return results


# --- 2. 딥러닝 모델 준비 및 예측 로직 ---
# 모델 정의
class CoffeeModel(nn.Module):
    # (FastAPI 버전과 동일)
    def __init__(self, n_countries, n_occupations, n_coffee_intakes):
        super().__init__()
        embed_dim = 10
        self.country_embed = nn.Embedding(n_countries, embed_dim)
        self.occupation_embed = nn.Embedding(n_occupations, embed_dim)
        self.coffee_embed = nn.Embedding(n_coffee_intakes, embed_dim)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 3 + 6),
            nn.Linear(embed_dim * 3 + 6, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        country = self.country_embed(x[:, 1].long())
        coffee = self.coffee_embed(x[:, 2].long())
        occupation = self.occupation_embed(x[:, 8].long())
        other_features = torch.cat([
            x[:, 0].unsqueeze(1), x[:, 3].unsqueeze(1), x[:, 4].unsqueeze(1),
            x[:, 5].unsqueeze(1), x[:, 6].unsqueeze(1), x[:, 7].unsqueeze(1)
        ], dim=1)
        x_combined = torch.cat([country, coffee, occupation, other_features], dim=1)
        return self.fc(x_combined)

# 모델 학습 및 로드 (싱글톤 패턴으로 서버 실행 시 한 번만 로드)
def get_trained_model():
    if not hasattr(get_trained_model, "model"):
        print("스트레스 관련 딥러닝 모델을 로드하고 학습합니다...")
        data = pd.read_csv('synthetic_coffee_health_10000.csv')
        # Drop unnecessary columns
        data = data.drop(['ID', 'Caffeine_mg', 'Sleep_Quality', 'Health_Issues', 'Heart_Rate'], axis=1)
        data = data[data['Gender'] != 'Other']
        data = pd.get_dummies(data, columns=['Gender'], drop_first=True, dtype=int)

        # Encode categorical columns
        for col in ['Country', 'Occupation', 'Coffee_Intake']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        # Encode Stress_Level
        le_stress = LabelEncoder()
        data['Stress_Level'] = le_stress.fit_transform(data['Stress_Level'])
        print("Stress_Level classes:", le_stress.classes_)  # Debug: Check encoded classes
        
        n_countries = data['Country'].max() + 1
        n_occupations = data['Occupation'].max() + 1
        n_coffee_intakes = data['Coffee_Intake'].max() + 1
        
        model = CoffeeModel(n_countries, n_occupations, n_coffee_intakes)
        
        X_dl = data.drop('Stress_Level', axis=1).values
        y_dl = data['Stress_Level'].values  # Now contains integers (e.g., 0, 1, 2, 3)
        X_train_dl, _, y_train_dl, _ = train_test_split(X_dl, y_dl, test_size=0.2, random_state=42)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            preds = model(torch.tensor(X_train_dl, dtype=torch.float))
            loss = criterion(preds, torch.tensor(y_train_dl, dtype=torch.long))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
        model.eval()
        get_trained_model.model = model
        get_trained_model.label_encoder = le_stress  # Store for prediction
        print("모델 학습 완료.")
    
    return get_trained_model.model

def predict_stress_category(input_data):
    model = get_trained_model()
    '''
    input_tensor = torch.tensor([[
        input_data['Age'], input_data['Coffee_Intake'],
        input_data['Sleep_Hours'], input_data['Physical_Activity_Hours'],
        input_data['Smoking'], input_data['Alcohol_Consumption'],
        input_data['Gender_Male'], input_data['Country'], input_data['Occupation']
    ]], dtype=torch.float)
    '''
    input_tensor = torch.tensor([[
        input_data['Age'], input_data['Country'], input_data['Coffee_Intake'],
        input_data['Sleep_Hours'], input_data['Physical_Activity_Hours'],
        input_data['Smoking'], input_data['Alcohol_Consumption'],
        input_data['Gender_Male'],  input_data['Occupation']
    ]], dtype=torch.float)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].numpy()
        predicted_class_index = torch.argmax(logits, dim=1).item()
        
    stress_categories = ['낮음(low)', '보통(Medium)', '높음H(igh)']
    predicted_category = stress_categories[predicted_class_index]
    
    return {
        "predicted_class": predicted_category,
        "predicted_index": predicted_class_index,
        "probabilities": {stress_categories[i]: f"{p:.4f}" for i, p in enumerate(probabilities)}
    }