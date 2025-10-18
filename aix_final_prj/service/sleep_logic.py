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
    
    le_country = LabelEncoder()
    le_occupation = LabelEncoder()
    le_coffee = LabelEncoder()
    le_gender = LabelEncoder()
    
    df1['Country'] = le_country.fit_transform(df1['Country'])
    df1['Occupation'] = le_occupation.fit_transform(df1['Occupation'])
    df1['Coffee_Intake'] = le_coffee.fit_transform(df1['Coffee_Intake'])
    df1['Gender'] = le_gender.fit_transform(df1['Gender'])
    
    df1['Poor_Sleep'] = (df1['Sleep_Hours'] < 6).astype(int)
# X와 y 설정
    X = df1[['Age',  'Country', 'Gender','Coffee_Intake', 'BMI', 'Physical_Activity_Hours', 'Smoking', 'Occupation', 'Alcohol_Consumption']]
    y = df1['Sleep_Hours']
    return df, df1, X, y, le_country, le_occupation, le_coffee, le_gender

# --- 1. 회귀 모델 분석 로직 ---
def run_regression_analysis():
    df, df1, X, y, le_country, le_occupation, le_coffee,le_gender = preprocess_data()
    feature_names = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(time.time()))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBRegressor(max_depth=5, random_state=42),
        'LightGBM': LGBMRegressor(max_depth=5, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        if name == 'RandomForest':
             model.fit(X_train, y_train)
             y_pred = model.predict(X_test)
             r2 = model.score(X_test, y_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = model.score(X_test_scaled, y_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        importances_dict = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            importances_dict = importance_df.to_dict('records')

        results[name] = {
            'RMSE': f"{rmse:.4f}",
            'R2': f"{r2:.4f}",
            'importances': importances_dict
        }
    return results

# --- 2. 분류 모델 분석 로직 ---
def run_classification_analysis():
    ddf, df1, X, y, le_country, le_occupation, le_coffee,le_gender = preprocess_data()
    y = df1['Sleep_Quality']
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

# --- 3. 군집 모델 분석 로직 ---
def run_clustering_analysis():
    df, df1, X, y, le_country, le_occupation, le_coffee,le_gender = preprocess_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(kmeans.inertia_)
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(diffs2) + 2
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    df1['Cluster'] = cluster_labels
    
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(X_scaled)
    df1["PCA1"] = pca_data[:, 0]
    df1["PCA2"] = pca_data[:, 1]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df1, palette="Set2", alpha=0.8)
    plt.title(f"KMeans Clustering (k={optimal_k}) with PCA", fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Django의 static 폴더에 저장
    if not os.path.exists("static/plots"):
        os.makedirs("static/plots")
    plot_filename = f"cluster_plot_{int(time.time())}.png"
    plot_path = os.path.join("static/plots", plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    cluster_analysis_results = []
    clustering_features = ['Age', 'BMI', 'Coffee_Intake','Physical_Activity_Hours']
    
    for cluster_id in range(optimal_k):
        cluster_data = df1[df1['Cluster'] == cluster_id]
        analysis = {
            "cluster_id": cluster_id,
            "count": len(cluster_data),
            "percentage": f"{len(cluster_data) / len(df1) * 100:.1f}%",
            "means": cluster_data[clustering_features].mean().round(2).to_dict()
        }
        cluster_analysis_results.append(analysis)

    return {
        "optimal_k": int(optimal_k),
        "plot_filename": "plots/" + plot_filename, # static 경로 내의 상대 경로
        "analysis": cluster_analysis_results
    }

# --- 4. 딥러닝 모델 준비 및 예측 로직 ---
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
            nn.Linear(16, 4)
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
        print("꿀잠 관련 딥러닝 모델을 로드하고 학습합니다...")
        data = pd.read_csv('synthetic_coffee_health_10000.csv')
        # (데이터 전처리 부분은 FastAPI 버전과 동일)
        data = data.dropna()
        data = data.drop(['ID', 'Caffeine_mg', 'Health_Issues', 'Heart_Rate'], axis=1)
        
        data = data[data['Gender'] != 'Other']
        data = pd.get_dummies(data, columns=['Gender'], drop_first=True, dtype=int)
        
        for col in ['Country','Occupation','Coffee_Intake']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        data["Stress_Level"] = pd.Categorical(
            data["Stress_Level"],
            categories=["Low", "Medium", "High"],
            ordered=True
        ).codes
        data = data[data["Stress_Level"] != -1]
        data["Sleep_Quality"] = pd.Categorical(
            data["Sleep_Quality"],
            categories=['Poor', 'Average', 'Good', 'Excellent'],
            ordered=True
        ).codes
        data = data[data["Sleep_Quality"] != -1]
        n_countries = data['Country'].max() + 1
        n_occupations = data['Occupation'].max() + 1
        n_coffee_intakes = data['Coffee_Intake'].max() + 1
        
        model = CoffeeModel(n_countries, n_occupations, n_coffee_intakes)
        
        X_dl = data.drop('Sleep_Quality', axis=1).values
        y_dl = data['Sleep_Quality'].values
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
        print("모델 학습 완료.")
    
    return get_trained_model.model

def predict_sleep_category(input_data):
    model = get_trained_model()
    input_tensor = torch.tensor([[
        input_data['Age'], input_data['Country'], input_data['Coffee_Intake'],
        input_data['Sleep_Hours'], input_data['BMI'],input_data['Gender_Male'], input_data['Physical_Activity_Hours'],input_data['Occupation'], input_data['Smoking'], input_data['Alcohol_Consumption']
    ]], dtype=torch.float)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].numpy()
        predicted_class_index = torch.argmax(logits, dim=1).item()
        
    sleep_categories = ['수면부족(Fair)', '평균(Average)', '좋음(Good)', '훌륭(Excellent)']
    predicted_category = sleep_categories[predicted_class_index]
    
    return {
        "predicted_class": predicted_category,
        "predicted_index": predicted_class_index,
        "probabilities": {sleep_categories[i]: f"{p:.4f}" for i, p in enumerate(probabilities)}
    }