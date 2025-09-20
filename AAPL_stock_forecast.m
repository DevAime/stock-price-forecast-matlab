%% Loading the 5 year AAPL stockprice dataset and Exploring it:
%Loding the AAPL dataset
data = readtable('AAPL.csv');
data
% Exploring the data set
head(data)
tail(data)
summary(data)

%% Handling missing values if any:
% Handlimg the dataset
missed = any(ismissing(data))
if missed
    disp("The missed values are:");
    disp(data(ismissing(data),:));
else
    disp("No missed values.");
end
% Filling with the previous data
data = fillmissing(data,'previous');

% Ensure the data is in chronological order
prices = flipud(data.Close);

%% Normalizing and calculating Indicators:

%Normalize or Scale the Data the price column for Time Series:
% Min- max Scalling:
data.Close = (data.Close - min(data.Close))/(max(data.Close) - min(data.Close));
data.Volume = (data.Volume - min(data.Volume))/(max(data.Volume) - min(data.Volume));
% Indicators movingAverage:
data.MovingAverage20 = movmean(data.Close,[19 0])
% 14 day RSI index:
data.RSI14 = rsindex(data.Close, 14);

%% Preparing the Data for Machine Learning:
% Format data for machine learning model
% Using  20 days past price to train and predict the current price
Numdays = 20;
x = [];
y = [];
for i = Numdays+1:height(data)
    x = [x;data.Close(i - Numdays:i-1)'];
    y = [y;data.Close(i)];
end

% Split the Data into Training and Test Sets: 
% Using 80% for training and 20% for testing:
train_ratio = 0.8;
num_train = floor(train_ratio*size(x,1));
X_train = x(1:num_train,:);
y_train = y(1:num_train);
x_test = x(num_train+1:end,:);
y_test = y(num_train+1:end);


%% Converting the the Data to time series:
% Convert data to time series format
dates = data.Date;
prices = data.Close;
ts = array2timetable(prices, 'RowTimes', dates);
ts

%% Implementation of Exponential Smoothing Time Series for Stock Price Prediction:

% Load the data from the CSV file
data = readtable('AAPL.csv');

% Convert the 'Date' column to datetime format
data.Date = datetime(data.Date, 'InputFormat', 'MM/dd/yyyy');

% Extract the 'Close' prices
prices = data.Close;

% Split data into training and test sets
trainRatio = 0.8;
numTrain = floor(trainRatio * length(prices));
trainData = prices(1:numTrain);
testData = prices(numTrain+1:end);

% Define the smoothing parameters for Holt-Winters method
alpha = 0.5; % Level smoothing parameter
beta = 0.3;  % Trend smoothing parameter
gamma = 0.1; % Seasonal smoothing parameter
seasonality = 12; % Assuming monthly seasonality (adjust as needed)

% Initialize the level, trend, and seasonal components
level = zeros(size(trainData));
trend = zeros(size(trainData));
seasonal = zeros(seasonality, 1); % Correctly size the seasonal component

% Initial values for level, trend, and seasonal components
level(1) = trainData(1);
trend(1) = trainData(2) - trainData(1);
for t = 1:seasonality
    seasonal(t) = trainData(t) - level(1);
end

% Apply Holt-Winters Exponential Smoothing
for t = 2:length(trainData)
    if t > seasonality
        level(t) = alpha * (trainData(t) - seasonal(mod(t-1, seasonality) + 1)) + (1 - alpha) * (level(t - 1) + trend(t - 1));
        trend(t) = beta * (level(t) - level(t - 1)) + (1 - beta) * trend(t - 1);
        seasonal(mod(t-1, seasonality) + 1) = gamma * (trainData(t) - level(t)) + (1 - gamma) * seasonal(mod(t-1, seasonality) + 1);
    else
        level(t) = alpha * (trainData(t) - seasonal(t)) + (1 - alpha) * (level(t - 1) + trend(t - 1));
        trend(t) = beta * (level(t) - level(t - 1)) + (1 - beta) * trend(t - 1);
        seasonal(t) = gamma * (trainData(t) - level(t)) + (1 - gamma) * seasonal(t);
    end
end

% Forecast the next 30 periods (with seasonality adjustment)
numForecastPeriods = 30;
forecastedPrices = zeros(numForecastPeriods, 1);

for i = 1:numForecastPeriods
    forecastedPrices(i) = level(end) + i * trend(end) + seasonal(mod(length(trainData) - seasonality + i, seasonality) + 1);
end

% Create a time vector for the forecasted dates
forecastDates = (data.Date(numTrain) + days(1:numForecastPeriods))';

% Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) using trainData
trainForecast = level(1:length(testData));  % Forecast for the test period based on training data

mse = mean((testData - trainForecast).^2);
mae = mean(abs(testData - trainForecast));

% Display performance metrics
fprintf('Validation Performance:\n');
fprintf('Mean Squared Error (MSE): %.4f\n', mse);
fprintf('Mean Absolute Error (MAE): %.4f\n', mae);

% Plot the results
figure;
hold on;
plot(data.Date, prices, 'b', 'DisplayName', 'Historical Prices');
plot(data.Date(1:numTrain), level, 'g', 'DisplayName', 'Smoothed Prices');
plot(forecastDates, forecastedPrices, 'r--', 'DisplayName', 'Forecasted Prices');
legend show;
xlabel('Date');
ylabel('Price');
title('AAPL Stock Price Forecast with Holt-Winters Exponential Smoothing');
hold off;

%%  The CNN Modified Implementation:
% Load the data
dataTable = readtable('AAPL.csv');

% Assuming the first column is the date and the rest are numerical data
dateColumn = dataTable{:, 1}; % Extract date column (if needed for reference)
data = dataTable{:, 2:end}; % Extract numerical data

% Normalize the data
mu = mean(data);
sigma = std(data);
data = (data - mu) ./ sigma;

% Create sequences and split data
sequenceLength = 30; % Number of past days to consider for prediction
numObservations = size(data, 1) - sequenceLength;
X = zeros(sequenceLength, size(data, 2), 1, numObservations); % 30 x 6 x 1 x numObservations
Y = zeros(numObservations, size(data, 2)); % numObservations x 6

for i = 1:numObservations
    X(:, :, 1, i) = data(i:i+sequenceLength-1, :); % 30 x 6 x 1 sequence
    Y(i, :) = data(i+sequenceLength, :); % Corresponding target value
end

% Split data into training and validation sets
trainRatio = 0.8;
numTrain = floor(numObservations * trainRatio);

XTrain = X(:, :, :, 1:numTrain);
YTrain = Y(1:numTrain, :);
XVal = X(:, :, :, numTrain+1:end);
YVal = Y(numTrain+1:end, :);

% Define CNN network architecture
inputSize = [sequenceLength, size(XTrain, 2), 1]; % 30 x 6 x 1
numFilters = 50;
filterSize = [10, size(XTrain, 2)]; % 10 x 6 filter size
outputSize = size(YTrain, 2); % Should be 6 (number of target features)

layers = [ ...
    imageInputLayer(inputSize, 'Normalization', 'none')
    convolution2dLayer(filterSize, numFilters, 'Padding', 'same')
    reluLayer
    fullyConnectedLayer(outputSize)
    regressionLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 50, ...
    'LearnRateDropFactor', 0.2, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', 0);

% Train the CNN network
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict and evaluate performance
YPred = predict(net, XVal);
YPred = YPred .* sigma + mu; % Revert normalization
YVal = YVal .* sigma + mu; % Revert normalization

% Evaluate performance
rmse = sqrt(mean((YPred - YVal).^2));
disp(['RMSE: ', num2str(rmse)]);

% Predict future values
XNew = data(end-sequenceLength+1:end, :);
XNew = reshape(XNew, [sequenceLength, size(data, 2), 1, 1]);

YPredNew = predict(net, XNew);
YPredNew = YPredNew .* sigma + mu; % Revert normalization

disp(['Predicted Value: ', num2str(YPredNew)]);

% Plot actual vs. predicted values
figure;
subplot(2,1,1);
plot(dateColumn(numTrain+sequenceLength+1:end), YVal(:, 4), '-');
hold on;
plot(dateColumn(numTrain+sequenceLength+1:end), YPred(:, 4), '-');
hold off;
title('Validation Data');
xlabel('Date');
ylabel('Stock Prices');
legend('Actual', 'Predicted');
grid on;

subplot(2,1,2);
plot(dateColumn, dataTable.Close, '-');
hold on;
futureDates = [dateColumn(end); dateColumn(end) + days(10)];
predictedClose = [dataTable.Close(end); YPredNew(4)];
plot(futureDates, predictedClose, '-');
hold off;
title('Actual vs. Predicted Future Value');
xlabel('Date');
ylabel('Stock Prices');
legend('Actual', 'Predicted');
grid on;