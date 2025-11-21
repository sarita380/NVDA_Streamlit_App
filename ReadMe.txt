NVDA Stock Price Prediction Dashboard (Streamlit + ML)
An interactive machine-learning dashboard that predicts NVIDIA (NVDA) daily closing prices using a combination of traditional ML, tree-based ensembles, and deep learning methods, alongside optional sentiment analysis features.
This project is deployed inside a Streamlit app and allows any user to upload a custom CSV, visualize historical prices, train multiple ML models, compare performance, and view predictions for the final test day.
 Project Features
1. Streamlit Interactive Dashboard
•	Upload your own NVDA CSV dataset
•	Auto-preview of raw and cleaned data
•	Configurable test-split and visualization window
•	Fully interactive charts, metrics, and model comparison tables
2. Machine Learning Models Implemented
The app trains a suite of models on the user-provided dataset:
Linear Regression
•	Serves as baseline predictive model
•	Performs well when relationships are linear
•	Very fast training time
RandomForest Regressor
•	Captures nonlinear relationships
•	Resistant to overfitting
•	Handles features like Volume, High/Low price ranges well
 XGBoost
•	Regularized gradient boosting
•	High performance on tabular financial data
•	Great for short-term price prediction tasks
LightGBM
•	Histogram-based boosting
•	Very fast, extremely scalable
•	Performs well on large feature sets
All models output:
•	Predicted vs. Actual charts
•	RMSE, MAE, and MAPE metrics
•	Last test-day prediction comparison
•	Model performance ranking

Automated Data Cleaning
The app automatically performs:
•	Date parsing
•	Sorting by chronological order
•	Handling missing values (forward-fill, back-fill, and zero-imputation fallback)
•	Validation of required columns
•	Feature extraction for model training

Supports datasets with:
•	Open, High, Low, Close, Volume
•	Optional: Sentiment (FinBERT/VADER scores)

Visualizations
The dashboard produces:
•	Dataset Preview
•	Price-over-time charts
•	Feature importance (tree-based models)
•	Side-by-side last-day predictions
•	Metric comparison table
•	Accuracy/performance badges

Tech Stack
Component	Technology
Front-end UI	Streamlit
Data Processing	Pandas, NumPy
Machine Learning	scikit-learn
Tree Ensembles	XGBoost, LightGBM
Visualization	Matplotlib, Streamlit Charts
Optional NLP	FinBERT, VADER sentiment scores

Workflow Inside the App
1.	Upload CSV
2.	Data cleaning & validation
3.	Generate technical indicators & features
4.	Time-series split (train/test)
5.	Train ML models
6.	Evaluate errors
7.	Rank models by performance
8.	Visualize predictions
Why This Project Matters
This project demonstrates:
•	Data engineering
•	Time-series preprocessing
•	Applying multiple ML paradigms
•	Ensemble learning
•	Model evaluation
•	UI/UX in Streamlit
•	Real-world financial data modeling
How to Run Locally
pip install -r requirements.txt
streamlit run app.py
Optional: Deploy to Streamlit Cloud
1.	Push this repo to GitHub
2.	Go to: https://share.streamlit.io
3.	Choose your repository
4.	Select app.py
5.	Deploy
Future Enhancements
•	Add an LSTM/GRU deep learning model
•	Use technical indicators (RSI, MACD, EMA)
•	Integrate FinBERT sentiment pipeline directly
•	Add hyperparameter tuning
•	Add SHAP interpretability dashboard
•	Deploy to public URL
