import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_save():
    # 1. Load dataset
    df = pd.read_csv('dataset_traffic_accident_prediction1.csv')
    
    # 2. Drop rows where target 'Accident' is missing
    df = df.dropna(subset=['Accident'])
    
    # 3. Exclude Accident_Severity (data leakage prevention) and target column
    X = df.drop(columns=['Accident', 'Accident_Severity'])
    y = df['Accident'].astype(int)
    
    num_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol', 'Driver_Age', 'Driver_Experience']
    cat_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
    
    # Save dataset unique category values for validation and defaults
    categories_meta = {}
    for col in cat_cols:
        categories_meta[col] = sorted([str(x) for x in df[col].dropna().unique()])
        
    num_meta = {}
    for col in num_cols:
        num_meta[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median())
        }
        
    with open('dataset_meta.json', 'w') as f:
        json.dump({
            'categorical': categories_meta,
            'numerical': num_meta,
            'total_records': len(df),
            'target_distribution': y.value_counts().to_dict()
        }, f, indent=2)
        
    # Preprocessor definition
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    }

    results = {}
    trained_pipelines = {}
    best_model_name = None
    best_score = -1.0

    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Calculate feature importances if available
        feature_names = num_cols.copy()
        onehot_encoder = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = onehot_encoder.get_feature_names_out(cat_cols).tolist()
        all_feature_names = feature_names + cat_feature_names
        
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(all_feature_names, [round(float(imp), 4) for imp in importances]))
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            feature_importance = dict(zip(all_feature_names, [round(float(imp), 4) for imp in importances]))

        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1': round(float(f1), 4),
            'confusion_matrix': cm,
            'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        trained_pipelines[name] = clf

        if f1 > best_score:
            best_score = f1
            best_model_name = name

    print(f"All 4 algorithms successfully trained! Best Model: {best_model_name} (F1: {best_score:.4f})")
    
    # Save ALL 4 pipelines so user can select any algorithm for prediction
    joblib.dump({
        'pipelines': trained_pipelines,
        'best_model': best_model_name,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }, 'all_models.pkl')
    
    # Save metrics
    with open('model_metrics.json', 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'metrics': results
        }, f, indent=2)
        
    print("Exported all_models.pkl and model_metrics.json.")

if __name__ == '__main__':
    train_and_save()
