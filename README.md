# Household Electricity Consumption Forecasting - Streamlit Application

A professional web application for predicting household electricity consumption using XGBoost machine learning model.
and this is the production demo:
https://household-electricity-consumption-forecasting-mwwxybtbkjgt7ddl.streamlit.app/
## Features

- **Single Prediction**: Predict energy consumption for a specific date and time
- **Batch Prediction**: Generate predictions for multiple days with visualizations
- **Model Analytics**: View model performance metrics and feature importance
- **Interactive Visualizations**: Plotly-based charts for time series, hourly patterns, and seasonal trends

## Project Structure

```
electricity-forecasting/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .streamlit/           # Streamlit configuration (optional)
│   └── config.toml       # App configuration
└── models/               # Model files (optional)
    └── xgb_model.pkl     # Trained XGBoost model
```

## Installation

### Local Setup

1. Clone or download the project files

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub repository

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Sign in with GitHub

4. Click "New app" and select your repository

5. Set the main file path to `app.py`

6. Click "Deploy"

### Heroku

1. Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

### Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. Build and run:
```bash
docker build -t electricity-forecast .
docker run -p 8501:8501 electricity-forecast
```

## Using the Application

### Single Prediction

1. Navigate to "Single Prediction" page
2. Select date and time
3. Enter previous readings (lag features)
4. Click "Predict Consumption"
5. View the predicted energy consumption and details

### Batch Prediction

1. Navigate to "Batch Prediction" page
2. Select start date and number of days
3. Enter average lag values
4. Click "Generate Predictions"
5. View charts and download predictions as CSV

### Model Analytics

1. Navigate to "Model Analytics" page
2. View performance metrics, feature importance, and model details

## Model Information

- **Algorithm**: XGBoost Regressor
- **Training Data**: December 2006 - August 2010
- **Performance**: MAE: 1.16 kW, MAPE: 18.09%

## Customization

Edit CSS, add features, or integrate your own trained model.

---

**Version**: 1.0.0
