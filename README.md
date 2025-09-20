# AAPL Stock Price Forecasting

This project aims to **forecast Apple (AAPL) stock prices** using historical data from the last **5 years**, retrieved from Yahoo Finance.  
We implemented two approaches to build predictive models:  

- **Time Series Forecasting Method:** Holt-Winters Exponential Smoothing for capturing level, trend, and seasonality.  
- **Deep Learning Method:** Convolutional Neural Network (CNN) applied to stock price sequences for pattern recognition and future prediction.  

---

## Project Structure
- **Data Preprocessing:** Handling missing values, normalization, and technical indicator calculation (moving average, RSI).  
- **Time Series Model:** Holt-Winters exponential smoothing for forecasting.  
- **Deep Learning Model:** CNN architecture trained on sliding windows of stock prices.  
- **Evaluation:** Performance metrics such as MSE, MAE, and RMSE are computed.  
- **Visualization:** Historical vs. predicted stock prices plotted for analysis.

- ## Results

### Holt-Winters Time Series Forecast
<img width="500px" alt="Screenshot 2025-09-20 214911" src="https://github.com/user-attachments/assets/ca4867ca-4fe4-4480-bf07-37a629bc5efb" />

### CNN Training Progress
<img width="500px" alt="Screenshot 2025-09-20 215006" src="https://github.com/user-attachments/assets/06769986-b9ef-42cf-b083-56cb94c7b80e" />


### CNN Validation & Forecast Results
<img width="500px" alt="Screenshot 2025-09-20 215029" src="https://github.com/user-attachments/assets/8a5d105d-ea4e-4980-816e-f8be75e43610" />

