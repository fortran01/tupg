import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('Appendix1_Revenue_Cost_activity.csv')

# Convert 'Total_Rev' and 'Total_COGS' to numeric type, removing dollar signs and commas
data['Total_Rev'] = pd.to_numeric(data['Total_Rev'].replace(
    '[$,]', '', regex=True), errors='coerce')
data['Total_COGS'] = pd.to_numeric(
    data['Total_COGS'].replace('[$,]', '', regex=True), errors='coerce')

# Now perform the calculation
data['Profitability'] = data['Total_Rev'] - data['Total_COGS'] - \
    (data['Ships'] * 7 + data['Orders'] * 0.17 + data['ExpOr'] * 267 +
     data['Queries'] * 33 + data['Design'] * 70)


print(data.head())

# Select features for the decision tree
features = ['Cor_Bo', 'Cor_Ca', 'Die_Bo', 'Ass_Ca', 'HD_Cor',
            'Ships', 'Orders', 'ExpOr', 'Queries', 'Design']

X = data[features]
y = data['Profitability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

if X_train.shape[0] == 0:
    raise ValueError("X_train is empty. Check your data splitting process.")

# Clean the data
X_train = X_train.replace(r'[$,]', '', regex=True).astype(float)
X_test = X_test.replace(r'[$,]', '', regex=True).astype(float)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

if X_train.shape[0] == 0:
    raise ValueError(
        "X_train is empty. Check your data cleaning process (handling of dollar signs)")

# Before scaling and fitting, handle NaN values in y
print("Before NaN handling:")
print("y_train shape:", y_train.shape)
print("X_train shape:", X_train.shape)
print("Number of NaN values in y_train:", y_train.isna().sum())

y_train = y_train.dropna()
X_train = X_train.loc[y_train.index]

print("After NaN handling:")
print("y_train shape:", y_train.shape)
print("X_train shape:", X_train.shape)

if X_train.shape[0] == 0:
    raise ValueError(
        "X_train is empty. Check your train data cleaning (handling of NaN values)")

# If you also have a test set, do the same for it
y_test = y_test.dropna()
X_test = X_test.loc[y_test.index]

print("Original dataset shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

if X_train.shape[0] == 0:
    raise ValueError(
        "X_train is empty. Check your test data cleaning process.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the decision tree model
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train_scaled, y_train)

# Get feature importances
feature_importance = dt_model.feature_importances_
feature_importance_dict = dict(zip(features, feature_importance))

# Sort features by importance
sorted_features = sorted(feature_importance_dict.items(),
                         key=lambda x: x[1], reverse=True)

print("Feature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Identify the top 5 drivers of customer profitability
top_5_drivers = sorted_features[:5]
print("\nTop 5 Drivers of Customer Profitability:")
for i, (feature, importance) in enumerate(top_5_drivers, 1):
    print(f"{i}. {feature}: {importance:.4f}")
