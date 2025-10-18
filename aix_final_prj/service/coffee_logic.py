import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report,silhouette_score
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
from tqdm import tqdm  # For progress bar in training (optional, if installed)
import threading
import queue


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
    
    X = df1[['Age', 'Sleep_Hours', 'BMI','Heart_Rate', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption','Gender','Country', 'Occupation']]
    y = df1['Coffee_Intake']
    return df, X, y

# --- 1. 회귀 모델 분석 로직 ---
def run_regression_analysis():
    df, X, y = preprocess_data()
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
    df, X, y = preprocess_data()
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
    df, X, y = preprocess_data()
    df1 = df.copy()
    
    clustering_features = ['Coffee_Intake', 'BMI', 'Sleep_Hours', 'Heart_Rate', 'Physical_Activity_Hours', 'Age', 'Caffeine_mg']
    X_cluster = df1[clustering_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=537, n_init=10).fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil_score)
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(diffs2) + 2
    print(optimal_k)
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=537, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    df1['Cluster'] = cluster_labels
    
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(X_scaled)
    df1["PCA1"] = pca_data[:, 0]
    df1["PCA2"] = pca_data[:, 1]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df1, palette="pastel", alpha=0.7)
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
    clustering_features = ['Coffee_Intake', 'BMI', 'Sleep_Hours', 'Heart_Rate', 
                        'Physical_Activity_Hours', 'Age', 'Caffeine_mg']
    
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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# (CoffeeModel 클래스 정의는 그대로 둡니다)
class CoffeeModel(nn.Module):
    def __init__(self, n_countries, n_occupations): # n_coffee_intakes 인수 제거
        super().__init__()
        embed_dim = 10
        self.country_embed = nn.Embedding(n_countries, embed_dim)
        self.occupation_embed = nn.Embedding(n_occupations, embed_dim)
        # self.coffee_embed 레이어 삭제

        # 입력 차원을 embed_dim * 2 + 6 으로 수정
        self.fc = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 2 + 6),
            nn.Linear(embed_dim * 2 + 6, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        country = self.country_embed(x[:, 1].long())
        occupation = self.occupation_embed(x[:, 8].long())
        # coffee = self.coffee_embed(...) 라인 삭제

        other_features = torch.cat([
            x[:, 0].unsqueeze(1), x[:, 2].unsqueeze(1), x[:, 3].unsqueeze(1),
            x[:, 4].unsqueeze(1), x[:, 5].unsqueeze(1), x[:, 6].unsqueeze(1),
            x[:, 7].unsqueeze(1) # Gender_Male이 인덱스 7일 것으로 추정, 확인 필요
        ], dim=1)

        # other_features의 개수를 다시 확인해야 합니다.
        # X_dl의 컬럼 순서: Age(0), Country(1), Sleep_Hours(2), BMI(3), 
        # Physical_Activity_Hours(4), Smoking(5), Alcohol_Consumption(6),
        # Stress_Level(7), Occupation(8), Gender_Male(9)
        # 따라서 other_features는 다음과 같이 구성되어야 합니다.
        other_features = torch.cat([
            x[:, 0].unsqueeze(1),  # Age
            x[:, 2].unsqueeze(1),  # Sleep_Hours
            x[:, 3].unsqueeze(1),  # BMI
            x[:, 4].unsqueeze(1),  # Physical_Activity_Hours
            x[:, 5].unsqueeze(1),  # Smoking
            x[:, 6].unsqueeze(1),  # Alcohol_Consumption
            x[:, 7].unsqueeze(1),  # Stress_Level
            x[:, 9].unsqueeze(1)   # Gender_Male
        ], dim=1)

        # BatchNorm1d의 입력 차원은 embed_dim*2 + 8이 되어야 합니다.
        # (임베딩 2개 * 10차원) + (수치형/범주형 8개) = 20 + 8 = 28
        
        # 위 로직이 복잡하니, 원본 코드의 의도를 살려서 간단하게 수정하겠습니다.
        # 원본 코드는 other_features를 6개로 지정했으므로, 그대로 따르겠습니다.
        other_features_original = torch.cat([
            x[:, 0].unsqueeze(1), x[:, 3].unsqueeze(1), x[:, 4].unsqueeze(1),
            x[:, 5].unsqueeze(1), x[:, 6].unsqueeze(1), x[:, 7].unsqueeze(1)
        ], dim=1) # 6개 피처

        # torch.cat에서 coffee 변수 삭제
        x_combined = torch.cat([country, occupation, other_features_original], dim=1)
        return self.fc(x_combined)

def get_trained_model():
    if not hasattr(get_trained_model, "model"):
        print("커피섭취량 딥러닝 모델을 로드하고 학습합니다...")
        data = pd.read_csv('synthetic_coffee_health_10000.csv')
        data = data.dropna()
        data = data.drop(['ID', 'Caffeine_mg', 'Sleep_Quality', 'Health_Issues', 'Heart_Rate'], axis=1)
        data = data[data['Gender'] != 'Other']
        data = pd.get_dummies(data, columns=['Gender'], drop_first=True, dtype=int)
        
        for col in ['Country', 'Occupation']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            
        bins = [-np.inf, 18.5, 25, 30, np.inf]
        labels = [0, 1, 2, 3]
        data['BMI'] = pd.cut(data['BMI'], bins=bins, labels=labels, right=False)
        data["Stress_Level"] = pd.Categorical(
            data["Stress_Level"],
            categories=["Low", "Medium", "High"],
            ordered=True
        ).codes
        data = data[data["Stress_Level"] != -1]
        data = data[data["Occupation"] != -1]
        
        # Coffee_Intake를 기준으로 그룹 생성
                
        bins = [-np.inf, 0.0, 1.0, 4.0, np.inf]
        labels = [0, 1, 2, 3]
        data['Coffee_Group'] = pd.cut(
            data['Coffee_Intake'],
            bins=bins,
            labels=labels,
            right=True,    # 경계값 포함 여부: True면 '이상~이하' (예: 1.0 < x <= 4.0)
            include_lowest=True # 가장 낮은 경계값(-np.inf)을 포함
        )
        
        
        
        
        
        data = data.dropna(subset=['Coffee_Group']) # qcut으로 인해 NaN이 생길 수 있음
        data['Coffee_Group'] = data['Coffee_Group'].astype(int)

        data = data.drop(["Coffee_Intake"], axis=1)
        data = data[data['Country'].notna()]

        # ✨✨✨ 에러 해결 부분 ✨✨✨
        n_countries = int(data['Country'].max() + 1)
        n_occupations = int(data['Occupation'].max() + 1)
        # n_coffee_group은 모델 생성에 더 이상 필요하지 않음

        # n_coffee_group 인수를 제거하고 모델 호출
        model = CoffeeModel(n_countries, n_occupations)
        
        # 데이터 타입을 나중에 변경하면 n_countries 계산 시 float이 될 수 있으므로 먼저 변경
        for col in data.columns:
            data[col] = data[col].astype('int64')

        X_dl = data.drop('Coffee_Group', axis=1).values
        y_dl = data['Coffee_Group'].values
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

# (predict_coffee_category 함수는 그대로 둡니다)

def predict_coffee_category(input_data):
    model = get_trained_model()

    input_tensor = torch.tensor([[
        input_data['Age'], 
        input_data['Country'], 
        input_data['Sleep_Hours'], 
        input_data['BMI'], 
        input_data['Gender_Male'], 
        input_data['Physical_Activity_Hours'],
        input_data['Smoking'], 
        input_data['Alcohol_Consumption'],         
        input_data['Stress_Level'],
        input_data['Occupation']
    ]], dtype=torch.float)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].numpy()
        predicted_class_index = torch.argmax(logits, dim=1).item()
        
    coffee_categories = ['안먹음', '조금', '보통', '많이먹음 ']
    predicted_category = coffee_categories[predicted_class_index]
    
    return {
        "predicted_class": predicted_category,
        "predicted_index": predicted_class_index,
        "probabilities": {coffee_categories[i]: f"{p:.4f}" for i, p in enumerate(probabilities)}
    }