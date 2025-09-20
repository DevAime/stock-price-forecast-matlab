# 📈 AAPL Stock Price Forecasting

This project aims to **forecast Apple (AAPL) stock prices** using historical data from the last **5 years**, retrieved from Yahoo Finance.  
We implemented two approaches to build predictive models:  

- **Time Series Forecasting Method:** Holt-Winters Exponential Smoothing for capturing level, trend, and seasonality.  
- **Deep Learning Method:** Convolutional Neural Network (CNN) applied to stock price sequences for pattern recognition and future prediction.  

---

## 🗂 Project Structure
- **Data Preprocessing:** Handling missing values, normalization, and technical indicator calculation (moving average, RSI).  
- **Time Series Model:** Holt-Winters exponential smoothing for forecasting.  
- **Deep Learning Model:** CNN architecture trained on sliding windows of stock prices.  
- **Evaluation:** Performance metrics such as MSE, MAE, and RMSE are computed.  
- **Visualization:** Historical vs. predicted stock prices plotted for analysis.

- ## 📊 Results

### Holt-Winters Time Series Forecast
<img src="images/holt_winters.png" width="500px">

### CNN Training Progress
<img src="images/cnn_training.png" width="500px">

### CNN Validation & Forecast Results
<img src="images/cnn_results.png" width="500px">
