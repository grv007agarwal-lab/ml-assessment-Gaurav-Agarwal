## B1. Problem Formulation

# (a) ML Problem Definition

** Target Variable
1. `items_sold` (continuous numerical value)

** Candidate Input Features
1. Store attributes: `store_id`, `store_size`, `location_type` (urban / semi-urban / rural)
2. Temporal features: `month`, `year`, `is_weekend`, `is_festival`, `is_month_end`
3. Promotion details: `promotion_type`
4. Market conditions: `competition_density`
5. (Optionally) lag features such as previous month sales if available

** Problem Type
1. Supervised Learning → Regression

** Justification
- The target `(items_sold)` is continuous
- The goal is to predict a numeric outcome based on historical labeled data
- Regression allows estimation of expected sales under different promotion scenarios, which can then be optimized for decision-making


# (b) Why Use Items Sold Instead of Revenue

** Issues with Revenue as Target

1. Revenue is influenced by price variations, discounts, and promotion mechanics
2. Different promotions (e.g., BOGO vs flat discount) distort revenue without reflecting true demand
3. High revenue does not necessarily mean high customer engagement or volume

** Advantages of Items Sold

1. Directly measures customer demand and promotion effectiveness
2. Comparable across different promotion types
3. Less sensitive to pricing strategies and discount structures

** Broader Principle

The target variable should align directly with the business objective and remain stable, unbiased, and interpretable.

** In real-world ML:

- Avoid targets influenced by external or confounding factors
- Prefer variables that reflect the core behavior you want to optimize


# (c) Alternative Modelling Strategy

A single global model assumes homogeneous behavior across stores, which is unrealistic.

** Proposed Strategy: Segment-Based or Hierarchical Modeling

** Option 1: Segmented Models

1. Build separate models for:
2. Urban stores
3. Semi-urban stores
4. Rural stores

** Option 2: Hierarchical / Mixed Modeling

1. Use a global model but include:
	- `store_id` as a feature
	-  Interaction terms (e.g., `promotion_type × location_type`)

** Justification

- Customer behavior and promotion response vary significantly by location
- Segmentation captures local patterns and heterogeneity
- Improves predictive accuracy and leads to better promotion decisions




## B2. Data and EDA Strategy

# (a) Data Joining, Grain, and Aggregation

** Join Strategy

Use a fact table + dimensions approach:

- Start with transactions (fact table)
- Join store attributes on `store_id`
- Join promotion details on `promotion_id` (or equivalent key)
- Join calendar on `transaction_date`

All joins are typically left joins from transactions, ensuring no loss of sales records.

** Grain of Final Dataset

One row = one store × one day (or one store × one month, depending on business frequency)

Recommended:

- If data is daily → store–day level
- If decision is monthly → aggregate to store–month level

** Aggregations Before Modelling

From transaction-level to modeling grain:

Target:
`items_sold = SUM(items)`

Promotion:
Encode dominant promotion per period (or use flags like % of days under each promotion)

Temporal:
Aggregate `is_weekend`, `is_festival` (e.g., count or proportion)

Store activity:
Total transactions (proxy for footfall)

Optional engineered features:
- Average basket size
- Lag features (previous period sales)


#(b) Exploratory Data Analysis (EDA)

** 1. Target Distribution (Histogram)
 What to check: Skewness, outliers in `items_sold`
 Impact:
	- Strong skew → consider log transformation
	- Outliers → may cap or treat separately

** 2. Promotion Type vs Items Sold (Boxplot)
 What to check: Sales variation across promotions
 Impact:
	- Identify high-performing promotions
	- May create interaction features (promotion × store type)

** 3. Time Series Plot (Sales over Time)
 What to check: Trends, seasonality, spikes (festivals, month-end)
 Impact:
	- Add features like `month`, `is_month_end`, lag variables
	- Detect non-stationarity

** 4. Correlation Heatmap (Numerical Features)
 What to check: Relationships between features and target
 Impact:
	- Identify strong predictors
	- Detect multicollinearity → may drop redundant features

** 5. Location Type / Store Size vs Sales (Bar or Boxplot)
 What to check: Differences in demand across segments
 Impact:
	- Justifies segmentation or interaction terms
	- Helps feature importance interpretation



# (c) Handling Promotion Imbalance

** Problem

 80% of data has no promotion
 Model may:
	- Learn to ignore promotion effects
	- Be biased toward predicting baseline (no-promo behavior)

** Mitigation Strategies

1. Rebalancing
	Oversample promotion periods or undersample non-promo data
	
2. Feature Engineering
	- Add binary flag: `is_promotion`
	- Create interaction features (promotion × store/location)
	
3. Model Choice
	Tree-based models (e.g., Random Forest, Gradient Boosting) handle imbalance better
	
4. Evaluation Strategy
	Evaluate separately on:
	- Promotion periods
	- Non-promotion periods
	


## B3. Model Evaluation and Deployment


#(c) End-to-End Deployment Process


** 1. Model Saving (Serialization)
Save the entire pipeline (preprocessing + model) to ensure consistency:
```
import joblib
joblib.dump(rf_pipeline, 'retail_model_pipeline.pkl')
```
Why:
	- Includes encoding, scaling, and model logic in one object
	- Prevents mismatch between training and inference transformations

** 2. Monthly Inference Workflow

At the start of each month:

Step 1: Prepare Input Data

Collect latest:
	- Store attributes (static)
	- Calendar features (month, festivals, weekends)
	- Planned promotion scenarios (one row per store × promotion option)

Step 2: Feature Engineering

Apply same transformations:
	- Date features (month, is_month_end, etc.)
	- Ensure schema matches training data exactly

Step 3: Load Model and Predict
	```
	model = joblib.load('retail_model_pipeline.pkl')
	predictions = model.predict(new_data)
	```
	
Step 4: Generate Recommendations

For each store:
	- Score all possible promotion types
	- Select promotion with highest predicted items_sold

Output: Recommended promotion per store for the month

** 3. Automation
Schedule as a monthly batch job (e.g., Airflow / cron)

Steps:
	1. Data ingestion
	2. Feature generation
	3.  Prediction
	4.  Store recommendations in database/dashboard


** 4. Monitoring & Performance Tracking

(a) Prediction Monitoring
Track:
	- Average predicted sales vs actual sales
	- Distribution of predictions over time

(b) Error Metrics (Post-Facto)
	- Compute periodically: RMSE, MAE on recent months
	- Compare with baseline training performance


** 5. Drift Detection

(a) Data Drift
	Monitor changes in input distributions:
		- Promotion frequency
		- Store traffic / competition
	Use statistical checks (e.g., distribution shifts)

(b) Concept Drift
	If relationship between features and target changes:
		Model errors increase even if input distribution is stable

** 6. Retraining Strategy

Trigger retraining when:

	- RMSE / MAE degrades beyond threshold (e.g., +10–15%)
	- Significant data drift detected
	- New promotion types or business changes introduced

Retraining Process

	- Append latest data
	- Rebuild pipeline
	- Validate on recent holdout period
	- Version and redeploy model
 