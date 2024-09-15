# TUPG Case Study

## Overview

This project analyzes customer profitability using a decision tree model. The main script, `decision_tree_model.py`, performs the following tasks:

1. Loads and preprocesses customer data from 'Appendix1_Revenue_Cost_activity.csv'.
2. Calculates customer profitability based on revenue, cost of goods sold, and activity-based costs.
3. Trains a decision tree regressor to predict customer profitability.
4. Identifies the top 5 drivers of customer profitability based on feature importance.

Key features of the analysis include:

- Data cleaning and handling of dollar-formatted values
- Splitting data into training and testing sets
- Scaling features using StandardScaler
- Training a DecisionTreeRegressor with a max depth of 5
- Calculating and ranking feature importances

The model provides insights into the most significant factors affecting customer profitability, which can be valuable for business decision-making and strategy development.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
