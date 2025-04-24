from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        
        # If no file selected, return error
        if file.filename == '':
            return "No file selected", 400
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load the dataset
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            return f"Error reading file: {e}", 400
        
        # Check if target column exists
        target_column = request.form.get('target_column')
        if target_column not in data.columns:
            return f"Target column '{target_column}' not found in dataset", 400
        
        # Split into features (X) and target (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Helper function to calculate MAPE
        def calculate_mape(actual, predicted):
            return round(np.mean(np.abs((actual - predicted) / actual)) * 100, 2) if np.any(actual != 0) else 0
        
        # Train and evaluate multiple models
        results = []
        
        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        mape_lr = calculate_mape(y_test, y_pred_lr)
        sample_lr = list(zip(y_test[:5], y_pred_lr[:5]))
        results.append({
            'model': 'Linear Regression',
            'r2': round(r2_lr, 4),
            'rmse': round(rmse_lr, 4),
            'mae': round(mae_lr, 4),
            'mape': mape_lr,
            'sample_predictions': sample_lr
        })
        
        # 2. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mape_rf = calculate_mape(y_test, y_pred_rf)
        sample_rf = list(zip(y_test[:5], y_pred_rf[:5]))
        results.append({
            'model': 'Random Forest',
            'r2': round(r2_rf, 4),
            'rmse': round(rmse_rf, 4),
            'mae': round(mae_rf, 4),
            'mape': mape_rf,
            'sample_predictions': sample_rf
        })
        
        # 3. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        mape_ridge = calculate_mape(y_test, y_pred_ridge)
        sample_ridge = list(zip(y_test[:5], y_pred_ridge[:5]))
        results.append({
            'model': 'Ridge Regression',
            'r2': round(r2_ridge, 4),
            'rmse': round(rmse_ridge, 4),
            'mae': round(mae_ridge, 4),
            'mape': mape_ridge,
            'sample_predictions': sample_ridge
        })
        
        # 4. Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        r2_lasso = r2_score(y_test, y_pred_lasso)
        rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
        mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
        mape_lasso = calculate_mape(y_test, y_pred_lasso)
        sample_lasso = list(zip(y_test[:5], y_pred_lasso[:5]))
        results.append({
            'model': 'Lasso Regression',
            'r2': round(r2_lasso, 4),
            'rmse': round(rmse_lasso, 4),
            'mae': round(mae_lasso, 4),
            'mape': mape_lasso,
            'sample_predictions': sample_lasso
        })
        
        # 5. Neural Network (MLP Regressor)
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        r2_mlp = r2_score(y_test, y_pred_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
        mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
        mape_mlp = calculate_mape(y_test, y_pred_mlp)
        sample_mlp = list(zip(y_test[:5], y_pred_mlp[:5]))
        results.append({
            'model': 'Neural Network (MLP)',
            'r2': round(r2_mlp, 4),
            'rmse': round(rmse_mlp, 4),
            'mae': round(mae_mlp, 4),
            'mape': mape_mlp,
            'sample_predictions': sample_mlp
        })
        
        # 6. K-Nearest Neighbors (KNN) Regressor
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        r2_knn = r2_score(y_test, y_pred_knn)
        rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
        mae_knn = mean_absolute_error(y_test, y_pred_knn)
        mape_knn = calculate_mape(y_test, y_pred_knn)
        sample_knn = list(zip(y_test[:5], y_pred_knn[:5]))
        results.append({
            'model': 'K-Nearest Neighbors (KNN)',
            'r2': round(r2_knn, 4),
            'rmse': round(rmse_knn, 4),
            'mae': round(mae_knn, 4),
            'mape': mape_knn,
            'sample_predictions': sample_knn
        })
        
        # 7. Decision Tree Regressor
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        r2_dt = r2_score(y_test, y_pred_dt)
        rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
        mae_dt = mean_absolute_error(y_test, y_pred_dt)
        mape_dt = calculate_mape(y_test, y_pred_dt)
        sample_dt = list(zip(y_test[:5], y_pred_dt[:5]))
        results.append({
            'model': 'Decision Tree',
            'r2': round(r2_dt, 4),
            'rmse': round(rmse_dt, 4),
            'mae': round(mae_dt, 4),
            'mape': mape_dt,
            'sample_predictions': sample_dt
        })
        
        # Prepare dataset preview
        dataset_preview = data.head().to_html(classes='table table-striped', index=False)
        
        # Prepare histogram data for predictions
        histograms = {}
        combined_actual = []
        combined_predicted = []
        model_labels = []

        for result in results:
            actual, predicted = zip(*result['sample_predictions'])  # Unpack sample predictions
            combined_actual.extend(actual)
            combined_predicted.extend(predicted)
            model_labels.extend([result['model']] * len(actual))  # Label each prediction with the model name
        
        # Combine all data into a single JSON object for the histogram
        histograms_combined = {
            'actual': combined_actual,
            'predicted': combined_predicted,
            'model_labels': model_labels
        }
        
        return render_template('index.html', dataset_preview=dataset_preview, results=results, histograms_combined=histograms_combined)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)