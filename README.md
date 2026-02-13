# HALF-HOURLY-SHORT-TERM-ELECTRICITY-DEMAND-FORECASTING-FOR-INDEPENDENT-SERVICE-OPERATORS
Accurate short-term electricity demand forecasting is critical for power system operation, generation dispatch, and market pricing.


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
