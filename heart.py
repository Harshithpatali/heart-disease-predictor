import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)

# Load the trained model and preprocessors
with open('svm_cardiovascular_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset to get feature names
def load_data():
    data = pd.read_csv(r'C:\Users\devar\Downloads\Heart disease\Cardiovascular_Disease_Dataset.csv')
    data = data[data['serumcholestrol'] >= 100]
    # Uncomment if zero values confirm need
    # data = data[data['restingBP'] >= 80]
    return data

# Generate plots (run once to populate static folder)
def generate_plots():
    df = load_data()
    X, y, df, _, _, _ = preprocess_data(df)

    
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.222, random_state=42, stratify=y_temp)
    
    # Slope-excluded model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_holdout_pred = model.predict(X_holdout)
    y_holdout_pred_proba = model.predict_proba(X_holdout)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    cm_test = confusion_matrix(y_test, y_pred)
    cm_holdout = confusion_matrix(y_holdout, y_holdout_pred)
    
    metrics = {
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Test ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Holdout Accuracy': accuracy_score(y_holdout, y_holdout_pred),
        'Holdout ROC AUC': roc_auc_score(y_holdout, y_holdout_pred_proba),
        '5-Fold CV Accuracy': 0.8957,
        '5-Fold CV Accuracy Std': 0.0275,
        'Confusion Matrix Test': cm_test,
        'Confusion Matrix Holdout': cm_holdout
    }
    
    plot_metrics_and_roc(metrics, fpr, tpr, df, model_name='Slope Excluded')
    
    # Ablated model
    X_train_ablated = X_train.drop(['noofmajorvessels', 'oldpeak'], axis=1, errors='ignore')
    X_test_ablated = X_test.drop(['noofmajorvessels', 'oldpeak'], axis=1, errors='ignore')
    X_holdout_ablated = X_holdout.drop(['noofmajorvessels', 'oldpeak'], axis=1, errors='ignore')
    
    ablated_model = SVC(probability=True, random_state=42, class_weight='balanced', C=0.1, gamma='scale', kernel='rbf')
    ablated_model.fit(X_train_ablated, y_train)
    
    y_pred_ablated = ablated_model.predict(X_test_ablated)
    y_pred_proba_ablated = ablated_model.predict_proba(X_test_ablated)[:, 1]
    y_holdout_pred_ablated = ablated_model.predict(X_holdout_ablated)
    y_holdout_pred_proba_ablated = ablated_model.predict_proba(X_holdout_ablated)[:, 1]
    
    fpr_ablated, tpr_ablated, _ = roc_curve(y_test, y_pred_proba_ablated)
    
    cm_test_ablated = confusion_matrix(y_test, y_pred_ablated)
    cm_holdout_ablated = confusion_matrix(y_holdout, y_holdout_pred_ablated)
    
    metrics_ablated = {
        'Test Accuracy': accuracy_score(y_test, y_pred_ablated),
        'Test ROC AUC': roc_auc_score(y_test, y_pred_proba_ablated),
        'Holdout Accuracy': accuracy_score(y_holdout, y_holdout_pred_ablated),
        'Holdout ROC AUC': roc_auc_score(y_holdout, y_holdout_pred_proba_ablated),
        'Confusion Matrix Test': cm_test_ablated,
        'Confusion Matrix Holdout': cm_holdout_ablated
    }
    
    plot_metrics_and_roc(metrics_ablated, fpr_ablated, tpr_ablated, df, model_name='Ablated Features')

# Preprocess data
def preprocess_data(df):
    X = df.drop(['patientid', 'target', 'slope'], axis=1, errors='ignore')
    y = df['target']
    
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, df, imputer, scaler, X.columns

# Plot metrics and ROC curves
def plot_metrics_and_roc(metrics, fpr, tpr, df, model_name='Slope Excluded'):
    os.makedirs('static', exist_ok=True)
    
    # Bar chart
    labels = ['Test', 'Holdout', '5-Fold CV']
    accuracy = [metrics['Test Accuracy'], metrics['Holdout Accuracy'], metrics.get('5-Fold CV Accuracy', 0.8957)]
    roc_auc = [metrics['Test ROC AUC'], metrics['Holdout ROC AUC']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x[:2] - width/2, accuracy[:2], width, label='Accuracy', color='skyblue')
    ax.bar(x[:2] + width/2, roc_auc, width, label='ROC AUC', color='lightcoral')
    ax.bar(x[2] - width/2, accuracy[2], width, color='skyblue')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    ax.set_title(f'SVM Model Performance Metrics ({model_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    ax.errorbar(x[2], accuracy[2], yerr=metrics.get('5-Fold CV Accuracy Std', 0.0275), fmt='none', c='black', capsize=5)
    
    plt.tight_layout()
    plt.savefig(f'static/svm_metrics_plot_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr if fpr is not None else [0, 1], tpr if tpr is not None else [0, 1], 
            label=f'ROC Curve (AUC = {metrics["Test ROC AUC"]:.4f})', color='skyblue')
    ax.plot([0, 1], [0, 1], label='Random Guess', linestyle='--', color='lightcoral')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'SVM ROC Curve (Test Set, {model_name})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'static/svm_roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Confusion matrix heatmaps
    for dataset, cm in [('Test', metrics['Confusion Matrix Test']), 
                        ('Holdout', metrics['Confusion Matrix Holdout'])]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix ({model_name} - {dataset})')
        plt.tight_layout()
        plt.savefig(f'static/confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset.lower()}.png')
        plt.close()
    
    # Correlation heatmap
    if model_name == 'Slope Excluded':
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df.drop(['patientid', 'target', 'slope'], axis=1, errors='ignore').corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Heatmap of Features (Slope Excluded)')
        plt.tight_layout()
        plt.savefig('static/correlation_heatmap.png')
        plt.close()

# Generate plots on startup
generate_plots()

# Load feature names
df = load_data()
_, _, _, _, _, feature_names = preprocess_data(df)

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {key: float(request.form[key]) for key in feature_names}
    
    # Prepare input
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Prepare results
    result = {
        'prediction': 'Positive' if prediction == 1 else 'Negative',
        'probability': f'{probability:.4f}',
        'metrics_slope_excluded': {
            '5-Fold CV Accuracy': '0.8957',
            '5-Fold CV Accuracy Std': '0.0275',
            'Test Accuracy': 'Unknown (awaiting metrics)',
            'Test ROC AUC': 'Unknown (awaiting metrics)',
            'Holdout Accuracy': 'Unknown (awaiting metrics)'
        }
    }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)