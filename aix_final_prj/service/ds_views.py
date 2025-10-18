# coffee_app/views.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from django.shortcuts import render

def coffee_analysis_view(request):
    # Set random seed and styling
    np.random.seed(537)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Load and preprocess data
    df = pd.read_csv('synthetic_coffee_health_10000.csv')  # Ensure the CSV is in the project directory
    df['Health_Issues'] = df['Health_Issues'].fillna('None')
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Coffee intake statistics
    coffee_stats = df['Coffee_Intake'].describe()
    coffee_summary = {
        'mean': f"{coffee_stats['mean']:.2f}",
        'median': f"{coffee_stats['50%']:.2f}",
        'std': f"{coffee_stats['std']:.2f}",
        'range': f"{coffee_stats['min']:.2f} - {coffee_stats['max']:.2f}",
        'iqr': f"{coffee_stats['75%'] - coffee_stats['25%']:.2f}"
    }

    # Health metrics
    health_metrics = {
        'bmi_mean': f"{df['BMI'].mean():.1f}",
        'sleep_mean': f"{df['Sleep_Hours'].mean():.1f}",
        'heart_rate_mean': f"{df['Heart_Rate'].mean():.0f}",
        'activity_mean': f"{df['Physical_Activity_Hours'].mean():.1f}"
    }

    # Categorical variables summary
    categorical_summary = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        most_common = df[col].value_counts().index[0]
        categorical_summary.append({
            'column': col,
            'unique_count': unique_count,
            'most_common': most_common
        })

    # Correlation analysis
    correlation_matrix = df[numerical_cols].corr()
    coffee_corrs = correlation_matrix['Coffee_Intake'].abs().sort_values(ascending=False)[1:].head(5)
    corr_summary = []
    for var, corr in coffee_corrs.items():
        actual_corr = correlation_matrix['Coffee_Intake'][var]
        direction = "↑" if actual_corr > 0 else "↓"
        corr_summary.append({
            'variable': var,
            'correlation': f"{actual_corr:.3f}",
            'direction': direction
        })

    # Country statistics
    country_stats = df.groupby('Country').agg({
        'Coffee_Intake': ['mean', 'std', 'count'],
        'BMI': 'mean',
        'Sleep_Hours': 'mean',
        'Heart_Rate': 'mean',
        'Physical_Activity_Hours': 'mean'
    }).round(2)
    country_stats.columns = ['Coffee_Mean', 'Coffee_Std', 'Sample_Size', 'BMI_Mean',
                            'Sleep_Mean', 'HeartRate_Mean', 'Activity_Mean']
    country_stats = country_stats.sort_values('Coffee_Mean', ascending=False)
    country_summary = [
        {'rank': idx+1, 'country': country, 'mean': row['Coffee_Mean'], 'sample_size': row['Sample_Size']}
        for idx, (country, row) in enumerate(country_stats.iterrows())
    ]

    # Gender analysis
    gender_summary = []
    for gender in df['Gender'].unique():
        gender_data = df[df['Gender'] == gender]
        gender_summary.append({
            'gender': gender,
            'count': len(gender_data),
            'coffee_mean': f"{gender_data['Coffee_Intake'].mean():.2f}",
            'coffee_std': f"{gender_data['Coffee_Intake'].std():.2f}",
            'age_mean': f"{gender_data['Age'].mean():.1f}",
            'bmi_mean': f"{gender_data['BMI'].mean():.1f}",
            'sleep_mean': f"{gender_data['Sleep_Hours'].mean():.1f}"
        })

    # Age group analysis
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100],
                            labels=['18-25세', '26-35세', '36-45세', '46-55세', '55세 이상'])
    age_group_stats = df.groupby('Age_Group')['Coffee_Intake'].agg(['mean', 'std', 'count']).round(2)
    age_summary = [
        {'age_group': age_group, 'mean': stats['mean'], 'std': stats['std'], 'count': stats['count']}
        for age_group, stats in age_group_stats.iterrows()
    ]

    # Encoding and ML preparation
    df_ml = df.copy()
    le_dict = {}
    encoded_cols = []
    encoding_summary = []
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_col = col + '_encoded'
        df_ml[encoded_col] = le.fit_transform(df_ml[col])
        le_dict[col] = le
        encoded_cols.append(encoded_col)
        encoding_summary.append({
            'column': col,
            'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
        })

    # Define feature sets
    features_for_coffee_prediction = ['Age', 'BMI', 'Sleep_Hours', 'Heart_Rate',
                                    'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption'] + encoded_cols
    features_for_health_prediction = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours',
                                    'BMI', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption']
    feature_summary = {
        'coffee_features_count': len(features_for_coffee_prediction),
        'health_features_count': len(features_for_health_prediction)
    }

    # Target variables
    df_ml['High_Coffee_Consumer'] = (df_ml['Coffee_Intake'] > df_ml['Coffee_Intake'].quantile(0.75)).astype(int)
    df_ml['Poor_Sleep'] = (df_ml['Sleep_Hours'] < 6).astype(int)
    df_ml['High_BMI'] = (df_ml['BMI'] >= 25).astype(int)
    target_summary = {
        'high_coffee': df_ml['High_Coffee_Consumer'].sum(),
        'high_coffee_pct': f"{df_ml['High_Coffee_Consumer'].mean()*100:.1f}",
        'poor_sleep': df_ml['Poor_Sleep'].sum(),
        'poor_sleep_pct': f"{df_ml['Poor_Sleep'].mean()*100:.1f}",
        'high_bmi': df_ml['High_BMI'].sum(),
        'high_bmi_pct': f"{df_ml['High_BMI'].mean()*100:.1f}"
    }

    # Check for multicollinearity
    feature_corr = df_ml[features_for_coffee_prediction].corr()
    high_corr_pairs = []
    for i in range(len(feature_corr.columns)):
        for j in range(i+1, len(feature_corr.columns)):
            if abs(feature_corr.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'var1': feature_corr.columns[i],
                    'var2': feature_corr.columns[j],
                    'corr': f"{feature_corr.iloc[i, j]:.3f}"
                })
    multicollinearity_summary = high_corr_pairs if high_corr_pairs else [{'message': '다중공산성이 없는 이슈 발견 (all |r| ≤ 0.8)'}]
    
    # Context for template
    context = {
        'coffee_summary': coffee_summary,
        'health_metrics': health_metrics,
        'categorical_summary': categorical_summary,
        'corr_summary': corr_summary,
        'country_summary': country_summary,
        'gender_summary': gender_summary,
        'age_summary': age_summary,
        'encoding_summary': encoding_summary,
        'feature_summary': feature_summary,
        'target_summary': target_summary,
        'multicollinearity_summary': multicollinearity_summary
    }

    return render(request, 'ds.html', context)