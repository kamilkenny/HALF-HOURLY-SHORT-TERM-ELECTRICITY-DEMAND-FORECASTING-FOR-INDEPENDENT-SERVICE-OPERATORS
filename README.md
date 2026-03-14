
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




![673bfd844d60be001dea9757 (1)](https://github.com/user-attachments/assets/24012a1c-f437-4844-abf8-78f0c0fe1018)


# GLOBAL RENEWABLE ENERGY ADOPTION: STRATEGIC ANALYSIS & COUNTRY-SPECIFIC ACTION PLAN

RESEARCHER - Kamil, Ridwan Kehinde

# Data Sources: 

World Energy Statistics (2000–2024), and IEA Energy Statistics Data



https://www.oecd.org/en/publications/world-energy-statistics_25183885.html

and 

https://www.iea.org/data-and-statistics/data-tools/energy-statistics-data-browser?country=WORLD&fuel=Energy%20supply&indicator=TESbySource





# Executive Summary
# This comprehensive analysis of 10 major economies reveals a pivotal moment in global renewable energy transition.

#  Aggregated Ouput Over time Considered in this Analysis

Total Energy Consumption (TWh)	

Per Capita Energy Use (kWh)	

Renewable Energy Share (%)	

Fossil Fuel Dependency (%)	

Industrial Energy Use (%)	

Household Energy Use (%)	

Carbon Emissions (Million Tons)	

Energy Price Index (USD/kW)

# COUNTRIES
	Russia
	China
	Brazil	
	Australia	
 	India	
    Canada	
	UK	
	Germany	
	USA	
	Japan 

# Total Countries Energy Consumption/Emmission/Share Trend Over Time


<img width="1201" height="701" alt="top countries by energy consumption" src="https://github.com/user-attachments/assets/4411af89-da03-4664-ae8f-106783ebfc3b" />


# Bar  Plot of Total Energy Consumption by Countries


<img width="591" height="509" alt="Bar  Plot of Total Energy Consumption by Countries" src="https://github.com/user-attachments/assets/cb94b141-0256-4cf0-b07e-b36e53b228c6" />

# Bar Plot of Per Capital Energy Use by Countries

<img width="601" height="509" alt="Bar Plot of Per Capital Energy Use by Countries" src="https://github.com/user-attachments/assets/70dc7c78-9d57-4df5-beab-8e2a121cf53a" />

# Bar Plot of Total Renewable Energy Share by Countries

<img width="572" height="509" alt="Bar Plot of Total Renewable Energy Share by Countries" src="https://github.com/user-attachments/assets/ec1b45cb-3f39-4be2-ba80-2cf06a25ae11" />

# Bar Plot of Total Fossil Fuel Share by Countries

<img width="572" height="509" alt="Bar Plot of Total Fossil Fuel Share by Countries" src="https://github.com/user-attachments/assets/85332b9e-4fbf-47af-8787-03d4e57c79a9" />


# Bar Plot of Total Carbon Emmision Intensity by Countries

<img width="577" height="509" alt="Bar Plot of Total Carbon Emmision Intensity by Countries" src="https://github.com/user-attachments/assets/ad16b54a-25a6-409a-b9ac-c6a615346bdc" />

# Bar Plot of Total Energy Price Index by Countries (USD/kW)

<img width="586" height="509" alt="Bar Plot of Total Energy Price Index by Countries" src="https://github.com/user-attachments/assets/b206f8a0-5a5e-41d8-affb-452114e2864f" />


# Energy use by Sectors
<img width="1184" height="584" alt="Energy use by sectors" src="https://github.com/user-attachments/assets/b89dae32-0fe8-4803-907e-a38ad190b418" />
<img width="1184" height="584" alt="Energy use by sectors 2" src="https://github.com/user-attachments/assets/7d80653b-44f3-45c7-989b-468ef7fde30e" />


# Energy Price Index by Countries

<img width="1183" height="928" alt="Energy Price Index by Countries" src="https://github.com/user-attachments/assets/140a4a25-33ac-4bf3-b2bb-f4bc859ccd54" />

# Total Carbon Emissions by Countries
<img width="1243" height="765" alt="Total Carbon Emissions by Countries" src="https://github.com/user-attachments/assets/929e0bb8-a1a9-418c-8f8d-3d85c9d70ce4" />

# Consumption/Mix Trends by Type - Global Average Over Time

<img width="1382" height="684" alt="Consumption Mix Trends by Type  Global Average Over Time" src="https://github.com/user-attachments/assets/47b48c04-394e-4a16-9459-34f00bb54f48" />

# Global Energy Mix (Average %) - Renewables vs Fossils

<img width="1382" height="684" alt="Global Energy Mix Re vs fossil" src="https://github.com/user-attachments/assets/d0fee5f5-f40e-4cc2-850a-a4863e6b032d" />


# Countries Driving the Drop in Total Energy TWh 

<img width="1184" height="584" alt="download" src="https://github.com/user-attachments/assets/e97fc8da-1269-459a-9b1b-a67ed0efc690" />




# RENEWABLE ENERGY TRENDS

# Fast Adopters of Renewable Energy by Countries
<img width="1990" height="1154" alt="Fast Adopter" src="https://github.com/user-attachments/assets/abbc13ba-7fcd-4ed2-8ddc-c6a8dd1cb334" />


# Renewable Energy Strategic Countries Insights 


<img width="1789" height="1180" alt="Re Strategic Countries Insights" src="https://github.com/user-attachments/assets/506afd73-0103-43c5-acc6-f39987206a0a" />

================================================================================
# 🎯 ACTIONABLE INSIGHTS
================================================================================

# 📈 OVERVIEW:
   • Total Countries Analyzed: 10
   • Fast-Growing Markets: 8 countries
   • High-Risk Countries: 1 countries
   • Leading & Growing: 3 countries
   • Emerging Successes: 1 countries

# 💡 IMMEDIATE ACTIONS:

# 🚀 INVESTMENT PRIORITIES (Fastest Growth):
   1. Brazil: +4.7% CAGR from 75.5% base
   2. Germany: +3.8% CAGR from 83.5% base
   3. Australia: +3.5% CAGR from 66.5% base

# 🆘 CRITICAL INTERVENTIONS (Highest Risk):
   1. Canada: -26.0%/yr decline at 26.2%

# 🏆 BEST PRACTICE SOURCES (Leaders + Momentum):
   1. Germany: 83.5% share, +7.4%/yr growth
   2. Russia: 71.6% share, +27.1%/yr growth
   3. Australia: 66.5% share, +12.9%/yr growth

# 🔭 EMERGING OPPORTUNITIES (Rapid Adoption):
   1. Canada: +3.3% CAGR, projected 29% in 3 years

================================================================================
# 📋 STRATEGIC RECOMMENDATIONS BY COUNTRY GROUP
================================================================================

# 📊 HIGH-GROWTH CHAMPIONS:
   Countries: Brazil, Germany, Australia, Canada, India, China, Russia, Japan
   → Action: Prioritize investment and market entry

# 📊 CRITICAL INTERVENTION NEEDED:
   Countries: Canada
   → Action: Urgent policy support and technical assistance

# 📊 BEST PRACTICE LEADERS:
   Countries: Germany, Russia, Australia
   → Action: Study and replicate successful policies

# 📊 EMERGING GROWTH MARKETS:
   Countries: Canada
   → Action: Monitor for future investment opportunities

================================================================================
# 🎯 COUNTRY-SPECIFIC STRATEGIC INSIGHTS
================================================================================

# 🚀 Highest Growth Potential:
   • Brazil: Invest now - +4.7% CAGR from 75.5% base
   • Germany: Invest now - +3.8% CAGR from 83.5% base

# 🆘 Most Urgent Intervention:
   • Canada: Critical support needed - -26.0%/yr decline

# 🏆 Best Practice Examples:
   • Germany: Learn from success - 83.5% share, still growing
   • Russia: Learn from success - 71.6% share, still growing

# 🔭 Emerging to Watch:
   • Canada: Monitor closely - +3.3% CAGR from low base

   # DEEP DIVE: FAST GROWING COUNTRIES IN RENEWABLE ENERGY ADOPTION

   <img width="1946" height="788" alt="DEEP DIVE FAST GROWING COUNTRIES IN RENEWABLE ENERGY ADOPTION" src="https://github.com/user-attachments/assets/632eced5-0c17-487c-b9c1-581611a5c48f" />

# 📊 FAST GROWING STARS ANALYSIS:
   • Average CAGR: 1.89%
   • Average Current Share: 58.3%
   • Growth Range: -2.28% to 4.73%

# 🎯 STRATEGIC OPPORTUNITIES:
   1. Brazil: +4.7% CAGR
      Current: 75.5% → Projected 3-year: 86.3%
   2. Germany: +3.8% CAGR
      Current: 83.5% → Projected 3-year: 93.1%
   3. Australia: +3.5% CAGR
      Current: 66.5% → Projected 3-year: 73.6%
   4. Canada: +3.3% CAGR
      Current: 26.2% → Projected 3-year: 28.7%

# DEEP DIVE: LEADERS WITH MOMENTUM

<img width="1998" height="784" alt="LEADERS WITH MOMENTUM" src="https://github.com/user-attachments/assets/36900b7a-f5f3-44f6-8489-32812006354b" />


# 📊 LEADERS WITH MOMENTUM ANALYSIS:
   • Average Renewable Share: 67.0%
   • Average Recent Growth: 12.90%/year
   • Highest Achiever: Germany (83.5%)

# 🎯 BEST PRACTICE AREAS:
   1. Germany:
      • 83.5% renewable share
      • +7.4%/year recent growth
      • +3.8% long-term CAGR
   2. Russia:
      • 71.6% renewable share
      • +27.1%/year recent growth
      • -0.9% long-term CAGR
   3. Australia:
      • 66.5% renewable share
      • +12.9%/year recent growth
      • +3.5% long-term CAGR

# DEEP DIVE: EMERGING SUCCESS STORIES
<img width="1948" height="788" alt="EMERGING SUCCESS STORIES" src="https://github.com/user-attachments/assets/2bfc8d85-707c-4477-849d-92adac1e71a5" />

# 📊 EMERGING SUCCESS STORIES ANALYSIS:
   • Average Starting Share: 12.1%
   • Average Current Share: 26.2%
   • Average CAGR: 3.26%

# 🎯 EMERGING OPPORTUNITIES:
   1. Canada:
      • Growth: 12.1% → 26.2%
      • +3.3% CAGR (2.2x growth multiplier)


# STRATEGIC COUNTRIES POSITIONING MATRIX

<img width="1591" height="1189" alt="STRATEGIC COUNTRIES POSITIONING MATRIX" src="https://github.com/user-attachments/assets/6d728f93-9664-4ebe-adbb-b574ddef6fa3" />
# 🎯 CREATING STRATEGIC POSITIONING MATRIX...

# 📈 Quadrant Distribution:
  • Champions: 3 countries (30.0%)
  • Laggards: 3 countries (30.0%)
  • Emerging Leaders: 2 countries (20.0%)
  • Established Leaders: 2 countries (20.0%)

# RENEWABLE ENERGY GROTH RATE OF COUNTRIES 

<img width="1387" height="989" alt="RENEWABLE ENERGY GROTH RATE OF COUNTRIES" src="https://github.com/user-attachments/assets/69b91137-e290-48a6-9c88-e51b596cabf8" />

# 🎯 KEY INSIGHTS FROM THIS VISUALIZATION:
# 📈 Top Performers:
   • Average CAGR: +0.42%
   • Average current renewable share: 52.0%
   • Growth range: -8.64% to 4.73%

# 📉 Bottom Performers:
   • Average CAGR: +0.42%
   • Average current renewable share: 52.0%
   • Growth range: -8.64% to 4.73%

# 🏆 STANDOUT COUNTRIES:
   • Fastest growing: Brazil (+4.73% CAGR)
   • Currently at: 75.5% renewable share
   • Slowest growing: UK (-8.64% CAGR)
   • Currently at: 6.9% renewable share
# CARBON INTENSITY ACROSS RENEWABLE-SHARE QUINTILES

<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/904850de-b98e-457b-8d78-91b24896972b" />


# DATA DISTRIBUTION/FREQUENCY OVER TIME 
Data Distribution of Years

<img width="585" height="455" alt="download" src="https://github.com/user-attachments/assets/608b6a79-4815-4a3e-a634-129c8cda9a3f" /> 
Data Distribution of Total Energy (twh)
<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/a0b9f7f4-43b7-4476-a53d-c8fc84bc9aac" />
Data Distribution of Per Capital (kwh)
<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/a0b9f7f4-43b7-4476-a53d-c8fc84bc9aac" />
Data Distribution of Per Capital (kwh)
<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/2b674357-4604-4a17-a5d9-124d188b1f1d" />
Data Distribution of Renewable Share
<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/96f6d9df-06a1-4bb7-bb2d-7542d7c38174" />
Data Distribution of Fossil Fuel Share
<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/9b12c4e4-9950-4150-a0a0-b5b55c87ffad" />
Data Distribution of Co2 Emmission 
<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/0c3915fa-9be4-4715-b99d-15f5e4e5e883" />

# ENERGY CONSUMPTION TRENDS BY TYPE (Individual Scales)

<img width="1584" height="1226" alt="download" src="https://github.com/user-attachments/assets/d8f40856-85f6-4492-9533-010e6213f849" />


# ENERGY PRICE INDEX BY COUNTRY 
<img width="1183" height="928" alt="download" src="https://github.com/user-attachments/assets/47bda18e-0c45-4fb0-a761-25726334977b" />

# ENERGY PRICE INDEX BY OVER TIME - GLOBAL AVERAGE WITH 5 YEARS MOVING AVERAGE 
<img width="1182" height="584" alt="download" src="https://github.com/user-attachments/assets/c7ab765d-43c4-4ddb-89d7-a20ad84e7829" />
<img width="1255" height="705" alt="GLOBAL AVERAGE WITH 5 YEARS MOVING AVERAGE" src="https://github.com/user-attachments/assets/7a1e92ca-3342-4ac2-8341-28e4379d21fb" />



