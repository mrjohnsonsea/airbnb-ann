# Georgetown MSBA
## OPAN 6603: Machine Learning II
### Project 2: Predicting Airbnb Prices with Artificial Neural Networks

**Authors:** Andrew Singh and Mike Johnson

---

## Overview

This project builds and evaluates Artificial Neural Network (ANN) models to predict Airbnb listing prices in Washington, D.C. using data scraped in July 2023 and made publicly available by [Inside Airbnb](http://insideairbnb.com/). The analysis also benchmarks ANN performance against linear regression models developed in a prior engagement.

---

## Dataset

- **Source:** Inside Airbnb
- **Listings:** 1,719 Airbnb listings in Washington, D.C. (scraped July 2023)
- **Features:** 15 characteristics per listing, including:
  - *Host statistics:* acceptance rate, total listings, start date, total reviews, average rating
  - *Listing characteristics:* bathrooms, bedrooms, beds, accommodates, minimum night stays, neighborhood, room type
  - *Target variable:* price

---

## Methodology

### Data Preprocessing
1. Performed exploratory data visualization to identify underlying trends
2. Removed irrelevant variables (e.g., `listing_id`)
3. Handled missing values via imputation:
   - `host_acceptance_rate`: imputed with training set mean
   - `bedrooms`: imputed using the average bedrooms per number of beds
4. Updated data types for categorical (factor), numeric, and date variables
5. Engineered `host_days` from `host_since`, representing days elapsed since the host's earliest listing date
6. Applied one-hot encoding for categorical variables (`neighborhood`, `room_type`)
7. Normalized all continuous features to [0, 1] range using min-max scaling

### Train / Test Split
- 70% training / 30% test split

### Models
Two ANN models were developed in R:

| Model | Package | Hidden Units | Weight Decay |
|-------|---------|-------------|-------------|
| caret | `caret` (`nnet` backend) | 2 | 0.01 |
| neuralnet | `neuralnet` | 2 | N/A |

The `caret` model hyperparameters (hidden units and weight decay) were selected via 10-fold cross-validation across a tuning grid. The `neuralnet` package does not support equivalent parameter tuning.

---

## Results

### ANN Model Performance (Test Set)

| Model | RMSE | R-Squared | MAE |
|-------|------|-----------|-----|
| caret | 0.8 | 0.54 | 0.5 |
| neuralnet | 0.8 | 0.53 | 0.5 |

The `caret` model achieved a moderate R-squared of 0.54, indicating it captures a meaningful but incomplete share of price variability. Both models performed similarly, with `caret` holding a slight edge.

### Comparison to Prior Linear Regression Models

| Model | R-Squared |
|-------|-----------|
| Multiple Linear Regression (bathrooms, accommodates, bedrooms) | 0.43 |
| Simple Linear Regression — Bathrooms | 0.38 |
| Simple Linear Regression — Accommodates | 0.34 |
| Simple Linear Regression — Bedrooms | 0.34 |

Both ANN models outperformed all prior regression models in predictive accuracy, though at the cost of interpretability. Regression models offer measurable marginal effects per variable; ANN models do not.

---

## Recommendation

The `caret` model is recommended for future use based on its slightly higher R-squared (0.54 vs. 0.53) and stronger predictive performance relative to the previously developed regression models.

---

## Repository Contents

| File | Description |
|------|-------------|
| `project_2_final.R` | Full R script: data preprocessing, EDA, model training, validation, and prediction |
| `AirbnbListings.csv` | Raw dataset (1,719 D.C. Airbnb listings, July 2023) |
| `Machine Learning II Project 2 - FINAL.pdf` | Written report with full methodology, findings, and appendix |

---

## Tools & Libraries

- **Language:** R
- `tidyverse`, `dplyr` — data manipulation
- `GGally` — exploratory visualization
- `caret` — model training and cross-validation
- `neuralnet` — alternative ANN implementation
- `fastDummies` — one-hot encoding
- `pROC` — ROC curve analysis

---

README written with the assistance of [Claude Code](https://claude.ai/code) by Anthropic.
