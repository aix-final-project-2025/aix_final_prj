def dsPreprocess():
    import numpy as np 
    import pandas as pd 
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv('synthetic_coffee_health_10000.csv')
    df['Health_Issues'] = df['Health_Issues'].fillna('None')
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    #print(categorical_cols)
    df_ml = df.copy()
    # Encode categorical variables

    for col in categorical_cols:
        df_ml[col] = df_ml[col].fillna('Unknown')
        
    le_dict = {}
    encoded_cols = []
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_col = col + '_encoded'
        df_ml[encoded_col] = le.fit_transform(df_ml[col]).astype(str)
        le_dict[col] = le
        encoded_cols.append(encoded_col)
        #print(f"인코딩 사항 {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
    df_ml = df_ml.drop(columns=categorical_cols)
    print(df_ml)
    return(df_ml)
