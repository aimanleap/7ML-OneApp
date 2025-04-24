# **7ML-OneApp**

Welcome to **7ML-OneApp**, a Flask-based web application that lets you train and evaluate **7 popular machine learning algorithms** on your own dataset—all in one place!

---

## **Features**
- Upload your CSV file and specify the target column.
- Train **7 regression models**:
  - Linear Regression
  - Random Forest
  - Ridge Regression
  - Lasso Regression
  - Neural Network (MLP)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- Evaluate model performance with metrics like R-squared, RMSE, MAE, and MAPE.
- Visualize predictions using histograms.
- Learn how each algorithm works with detailed explanations.

---

## **Installation**

### **Prerequisites**
- Python 3.8 or higher
- Git

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/AiManLeap/7ML-OneApp.git
   cd 7ML-OneApp


2 Install dependencies: 
    
    pip install -r requirements.txt
 

Run the app: 

    python app.py
     

Open your browser and navigate to http://127.0.0.1:5000. 
     

### Usage  

    Upload your CSV file.
    Specify the target column name.
    Click "Train Models" to see the results.
      
### Project **Structure**  
7ML-OneApp/
│
├── app.py                     # Flask application file
├── requirements.txt            # List of Python dependencies
├── README.md                   # Project description and instructions
├── static/                     # Static files (CSS, JS, etc.)
│   └── styles.css              # Custom CSS file
├── templates/                  # HTML templates
│   └── index.html              # Main HTML template
├── uploads/                    # Folder for uploaded files (ignored in .gitignore)
└── .gitignore                  # Specifies files/folders to ignore


### Contributing  

Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback is always welcome! 

### License  

This project is licensed under the MIT License.

### Made with ❤️ by AiManLeap 