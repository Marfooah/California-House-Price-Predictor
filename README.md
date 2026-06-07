# 🏠 California Housing Price Predictor

## 📌 Overview

This project is an interactive machine learning web application that predicts California housing prices using an **Artificial Neural Network (ANN)** built with **Scikit-learn's MLPRegressor** and deployed using **Streamlit**.

🚀 Live Demo: https://california-house-price-predictorr.streamlit.app/

📂 Dataset: California Housing Dataset

🧠 Model: Artificial Neural Network (MLPRegressor)

🎯 Test R² Score: 0.791

## 📸 Application Preview

<img width="2956" height="1648" alt="image" src="https://github.com/user-attachments/assets/afc5af81-1708-4a63-9b7e-052cfc4948a5" />

<img width="2956" height="1648" alt="image" src="https://github.com/user-attachments/assets/5f37b84c-0456-4d7f-8a20-40a3104ef03e" />

<img width="2956" height="1008" alt="image" src="https://github.com/user-attachments/assets/a99ea094-3f68-466e-906d-7b6c9f550d3f" />


The application allows users to:

* Predict median house values based on housing characteristics.
* Explore model performance metrics.
* Visualize training behavior and prediction quality.
* Analyze the California Housing dataset interactively.
* Explore housing prices geographically through interactive maps.
* Understand the architecture and configuration of the neural network.

---

## 🚀 Live Features

### 🎯 House Price Prediction

Users can enter housing features and receive an instant prediction of the median house value.

Input Features:

* Median Income
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

---

### 📉 Model Performance Dashboard

The application displays key regression metrics:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

These metrics help evaluate how well the model generalizes to unseen data.

---

### 📊 Dataset Explorer

Explore the California Housing dataset through:

* Dataset preview
* Descriptive statistics
* Feature distributions
* Correlation heatmaps
* Feature vs target analysis

---

### 🗺️ Interactive California Housing Map

Visualize housing values geographically using:

* Scatter Map Visualization
* Density Heatmap

This provides spatial insights into housing prices across California.

---

### 🏗️ Neural Network Architecture Viewer

The app dynamically displays:

* Input Layer
* Hidden Layer 1
* Hidden Layer 2
* Output Layer

Users can also modify:

* Number of neurons
* Learning rate
* Maximum iterations

to experiment with model performance.

---

## 🧠 Machine Learning Model

### Model Type

Artificial Neural Network (ANN)

Implemented using:

```python
MLPRegressor
```

### Architecture

Default configuration:

```text
Input Layer: 8 Features
Hidden Layer 1: 256 Neurons
Hidden Layer 2: 128 Neurons
Output Layer: 1 Neuron
```

### Activation Function

```text
ReLU
```

### Optimizer

```text
Adam
```

### Loss Function

```text
Mean Squared Error (MSE)
```

### Feature Scaling

```text
StandardScaler
```

All input features are standardized before training.

---

## 📈 Model Performance

Final Test Results:

| Metric   | Score  |
| -------- | ------ |
| R² Score | 0.791  |
| MSE      | ~0.264 |
| MAE      | ~0.343 |
| RMSE     | ~0.514 |

### Interpretation

* The model explains approximately **79.1% of the variance** in California housing prices.
* Average prediction error is approximately **$34,000**.
* Typical prediction deviation is approximately **$51,000**.

---

## 📂 Dataset

Dataset Source:

```python
sklearn.datasets.fetch_california_housing()
```

Dataset Statistics:

* 20,640 housing records
* 8 input features
* Continuous target variable

Target:

```text
Median House Value
```

Values are represented in units of $100,000.

---

## 🛠️ Tech Stack

### Machine Learning

* Scikit-learn
* NumPy
* Pandas

### Data Visualization

* Plotly
* Matplotlib

### Deployment

* Streamlit

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
├── app.py
├── requirements.txt
├── California_Housing_ANN.ipynb
├── README.md
└── assets/
    └── screenshots
```

---

## 🎯 Learning Outcomes

This project demonstrates:

* Data preprocessing
* Feature scaling
* Neural network implementation
* Regression modeling
* Hyperparameter tuning
* Model evaluation
* Interactive dashboard development
* Machine learning deployment using Streamlit

---

## 🔮 Future Improvements

Potential enhancements include:

* Hyperparameter optimization
* Model comparison with Random Forest and XGBoost
* Feature importance analysis
* Model persistence using joblib
* Real-time cloud deployment
* Advanced explainability visualizations

---

## 👩‍💻 Author

**Ayesha Tariq**

Artificial Intelligence Student | Machine Learning Enthusiast

Passionate about building AI-powered applications that transform data into meaningful insights.

---

## ⭐ If you found this project interesting

Consider giving the repository a star and sharing your feedback.
