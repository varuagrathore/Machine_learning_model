# Machine_learning_model
this repo contains all my work through the journey of machine learning excluded machine learning projects kaggle competitions notebooks



# 🏠 California Housing Prices Analysis

Welcome to the California Housing Prices Analysis project! This repository contains the code and data for analyzing housing prices in California. The project includes data preprocessing, visualization, model training, and evaluation.

## 📂 Project Structure

- `California Housing Prices.ipynb`: Jupyter notebook containing the entire analysis.
- `data/`: [Directory containing the dataset used for analysis.](https://www.kaggle.com/datasets/camnugent/california-housing-prices?select=housing.csv)
- `models/`: Directory containing the trained models.
- `README.md`: This file, providing an overview of the project.

## 📊 Project Overview

### 1. Data Preprocessing
- **Loading Data**: The housing dataset is loaded and inspected for initial understanding.
- **Handling Missing Values**: Missing values are handled using imputation techniques.
- **Feature Engineering**: New features are created to enhance the model's predictive power.
- **Data Splitting**: The data is split into training and testing sets using stratified sampling to ensure representative samples.

### 2. Data Visualization
- **Scatter Plots**: Visualize the geographical distribution of housing prices.
- **Correlation Matrix**: Understand the relationships between different features.
- **Trend Lines**: Identify trends and patterns in the data.

### 3. Model Training
- **Random Forest Regressor**: Train a Random Forest model to predict housing prices.
- **Model Evaluation**: Evaluate the model's performance using RMSE (Root Mean Squared Error).

### 4. Model Saving and Loading
- **Joblib**: Save and load the trained model using Joblib for efficient storage and retrieval.

## 🚀 Getting Started

### Prerequisites
- Python 3.6 or higher
- Jupyter Notebook
- Git
- Git LFS

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/varuagrathore/Machine_learning_model.git
   cd Machine_learning_model
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Git LFS**
   Ensure you have Git LFS installed to handle large files:
   ```bash
   choco install git-lfs  # For Windows
   brew install git-lfs   # For macOS
   sudo apt-get install git-lfs  # For Linux
   git lfs install
   ```

### Usage
1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
2. **Run the Notebook**
   Open `California Housing Prices.ipynb` and run all cells to execute the analysis.

## 🔍 Analyzing the Data
The notebook provides a comprehensive analysis, including data visualization, feature engineering, and model training. Key insights include:
- **Location-Based Pricing**: Housing prices are significantly influenced by location, particularly proximity to the ocean.
- **Population Density**: High population density areas tend to have higher housing prices.

## 📈 Model Performance
The Random Forest model achieved an RMSE of `18677.42`, indicating a reasonable performance for predicting housing prices.

## 💾 Handling Large Files with Git LFS
To manage large files like the trained model (`forest_reg.pkl`), Git LFS is used. Here's how to track and push large files:

1. **Track the Large File**
   ```bash
   git lfs track "models/forest_reg.pkl"
   ```

2. **Add and Commit**
   ```bash
   git add .gitattributes
   git add "models/forest_reg.pkl"
   git commit -m "Add forest_reg.pkl using Git LFS"
   ```

3. **Push to Remote Repository**
   ```bash
   git push origin main
   ```




