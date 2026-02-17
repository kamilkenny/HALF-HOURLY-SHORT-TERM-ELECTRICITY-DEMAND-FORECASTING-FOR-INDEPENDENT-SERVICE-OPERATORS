
# HALF-HOURLY-SHORT-TERM-ELECTRICITY-DEMAND-FORECASTING-FOR-INDEPENDENT-SERVICE-OPERATORS

![Project Banner](https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?q=80&w=1000)



Accurate short-term electricity demand forecasting is critical for power system operation, generation dispatch, and market pricing.

# The Design, Development and Modelling was done in Python and uploaded as a file in this respository 

# Plot Total Demand over January
<img width="1255" height="470" alt="download (3)" src="https://github.com/user-attachments/assets/32ff1c6d-cb15-4caf-8984-0d983f7589a7" />

# Average Demand by Day of Week

<img width="1025" height="479" alt="download (10)" src="https://github.com/user-attachments/assets/6c1ef46b-82ae-482c-88e8-286928782dd0" />


# Plot RRP (Price) over time

<img width="1238" height="470" alt="download (4)" src="https://github.com/user-attachments/assets/7f93f4f1-6290-476d-a7db-ca9f18fdf1c6" />
# Average Spot Price (RRP) by Day of the Week
<img width="1006" height="479" alt="download (11)" src="https://github.com/user-attachments/assets/e823d9ca-6680-4d5f-af88-c83eed0e5f4d" />


# Daily aggregate
Sum half-hourly demand per day to see daily patterns

<img width="1031" height="470" alt="download (5)" src="https://github.com/user-attachments/assets/7bbc2764-abfb-4d94-9f49-8f1da08ecf97" />

# Distribution of Demand & Price
Understand frequency and spread of values
<img width="1005" height="470" alt="download" src="https://github.com/user-attachments/assets/d296dca6-969f-42d0-a06e-dccd6762581e" />
<img width="1005" height="470" alt="download" src="https://github.com/user-attachments/assets/379ca555-02bc-42db-83d6-99c678c22046" />

# Boxplot to check daily variation
Helps identify peak load hours.
<img width="1023" height="470" alt="download (8)" src="https://github.com/user-attachments/assets/805ad210-a748-400c-b045-d29b864cf35f" />

# Correlation between Demand and Price
Checks if high demand corresponds to high prices.
<img width="618" height="470" alt="download (9)" src="https://github.com/user-attachments/assets/755b3dea-4cab-4825-88bc-6a943281419b" />

# Half-Hourly Forecasting Feature Pipeline
This pipeline transforms raw half-hourly energy demand data into a structured, model-ready dataset for predicting the next 3 half-hour demand values. It creates lag features, rolling statistics, time-based encodings, and multi-step targets.

# Steps in the Pipeline
Initialize feature DataFrame

Create an empty DataFrame with the same timestamps as the input data to store all engineered features.

Lag features

Add previous half-hour values of energy demand (e.g., lag_1, lag_2, lag_3).

Purpose: Captures temporal dependencies since past demand helps predict future demand.

Rolling statistics

Compute rolling mean and standard deviation over a window of recent half-hours.

Purpose: Captures short-term trends and variability in demand.

Hour of the day (cyclical encoding)

Convert the hour into hour_sin and hour_cos using sine and cosine transformations.

Purpose: Models the daily demand cycle and avoids discontinuities between 23:30 → 00:00.

Day of the week

Encode weekdays as binary columns using one-hot encoding.

Purpose: Accounts for different patterns between weekdays and weekends.

# Targets for multi-step forecasting

Create target_1, target_2, target_3 as the demand for the next 1, 2, and 3 half-hours.

Purpose: Prepares the dataset for predicting multiple future time steps simultaneously.

Drop rows with missing values

Remove rows that have NaN due to lag or target shifts.

Purpose: Ensures the dataset is complete for model training.

Output

Features: Lag values, rolling stats, hour/day encodings

Targets: Next 3 half-hour energy demands (target_1, target_2, target_3)

# Result: A clean, structured dataset ready for machine learning models, supporting multi-step forecasting.
# This pipeline efficiently captures temporal trends, cyclical patterns, and short-term variability, making it ideal for half-hourly energy demand forecasting.

<img width="1536" height="1024" alt="Image Feb" src="https://github.com/user-attachments/assets/faa1c52e-fc4c-4987-8996-47bbe47dc013" />

# Preparing dataset for predicting the next 3 half-hour energy demand values (target_1, target_2, target_3) by creating features that capture recent trends, daily/weekly seasonality, and temporal patterns.
# Step 1: Initialize DataFrame
# Create a new DataFrame batch_df using the timestamps from the original data.

This will hold all engineered features for the model.

# Step 2: Lag Features
Add previous values of the target variable as features:

lag_1 → 1 half-hour ago

lag_2 → 2 half-hours ago

lag_3 → 3 half-hours ago

Captures short-term trends in energy demand.

# Step 3: Rolling Statistics
Compute rolling mean and rolling standard deviation over the last rolling_window half-hours:

rolling_mean_3 → average of past 3 intervals

rolling_std_3 → standard deviation of past 3 intervals

Captures short-term trend and volatility in demand.

# Step 4: Cyclical Hour Features
Extract the hour of the day from the timestamp.

Encode it cyclically using sine and cosine:

hour_sin = sin(2π * hour / 24) hour_cos = cos(2π * hour / 24)

Captures daily patterns (e.g., morning/evening peaks).

# Step 5: Day-of-Week Features
Extract day of the week (0=Monday, 6=Sunday).

One-hot encode as separate columns (day_of_week_1, day_of_week_2, …), dropping the first to avoid multicollinearity.

Captures weekly seasonality (weekdays vs weekends).

# Step 6: Multi-Step Targets
Shift the target column to create the next 3 half-hour predictions:

target_1 → next half-hour

target_2 → 1 hour ahead

target_3 → 1.5 hours ahead

Ensures features at the current timestamp correspond to future demand.

# Step 7: Clean Data
Drop all rows with missing values from lagging, rolling, or shifted target operations.

Output is a clean, model-ready DataFrame.

Output

batch_df contains:

Lag features (lag_1, lag_2, lag_3)

Rolling statistics (rolling_mean_3, rolling_std_3)

Cyclical hour features (hour_sin, hour_cos)

Day-of-week one-hot encoded features

Multi-step targets (target_1, target_2, target_3)

<img width="1024" height="1536" alt="Image" src="https://github.com/user-attachments/assets/8a160181-161e-4a70-9d63-5f4183ca49d9" />
<img width="1214" height="556" alt="download (12)" src="https://github.com/user-attachments/assets/612bf04b-a858-4064-8ebe-cf0d3734f876" />


# Half-Hourly Future Forecasting Pipeline
This pipeline allows you to forecast the next 3 half-hour steps of electricity demand using previously trained XGBoost models and a recursive feature update approach.

# 1. Load Models
Load the trained models for each target (target_1, target_2, target_3) using joblib.

Each model predicts demand for the corresponding future half-hour step.

# 2. Define Forecast Horizon
Decide how many future half-hours to predict (n_steps = 3).

Generate future timestamps starting 30 minutes after the last data point. This ensures predictions are truly beyond the existing dataset.

# 3. Prepare Initial Features
Take the last row of features from your training data (latest_features) to initialize the forecasting process.

These features include lagged demands, rolling statistics, cyclical time features (hour sine/cosine), and day-of-week dummies.

# 4. Recursive Forecasting Loop
For each future half-hour step:

Predict demand using the model for each target.

Store the predicted values in a results table (future_preds).

Update lag features for the next prediction step:

lag_1 becomes the most recent predicted demand.

lag_2 takes the previous lag_1, lag_3 takes the previous lag_2, etc.

This ensures that the model can use its own predictions as inputs for subsequent steps, allowing multi-step forecasting.

# 5. Build Forecast Table
Combine the generated future timestamps with the predicted values for all three targets.

The resulting table is structured with columns
<img width="1024" height="1536" alt="f0641be7-5439-4f6b-b746-86bb7e567b3d" src="https://github.com/user-attachments/assets/e2c166e0-6bd1-4b04-a473-3e04d5157fc9" />
<img width="984" height="584" alt="download (13)" src="https://github.com/user-attachments/assets/48fbcd9c-59cc-4d43-8550-d23c9030219e" />

<img width="1189" height="556" alt="download" src="https://github.com/user-attachments/assets/df4999a8-95b4-4b99-a090-16f7470bed85" />


