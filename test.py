import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from category_encoders import TargetEncoder
from matplotlib import pyplot
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv('sample.csv')

# Impute missing values
imputer = SimpleImputer(strategy='median')
df['sales'] = imputer.fit_transform(df[['sales']])
df['competitor_price'] = imputer.fit_transform(df[['competitor_price']])

# Calculate price elasticity
df['price_elasticity'] = df['sales'] / df['competitor_price']

# Drop unnecessary columns
df.drop(['order_id', 'product_id', 'ship_date', 'customer_name', 'state', 'order_priority', 'year', 
         'country', 'product_name', 'order_date', 'ship_mode', 'market', 'region', 'sub_category'], 
        axis=1, inplace=True)

# Categorical features to be encoded
categorical_features = ['segment', 'category']

# Fill missing values in categorical columns
df[categorical_features] = df[categorical_features].fillna('Unknown')

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_features]), 
                                columns=encoder.get_feature_names_out(categorical_features))

# Ensure that indices align between the original and encoded dataframes
encoded_features.index = df.index

# Concatenate the original data with the encoded features
df = pd.concat([df, encoded_features], axis=1)

# Drop the original categorical columns after encoding
df.drop(columns=categorical_features, inplace=True)

# Add a new 'cost price' column based on sales and profit
df['cost price'] = df['sales'] - df['profit']

# Convert discount from percentage to actual discount
df['discount'] = df['sales'] * df['discount']

# Define features (X) and target variable (y)
X = df.drop('sales', axis=1)
y = df['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Add a 'Predicted Price' column based on model predictions
df['Predicted Price'] = model.predict(X)

# Define profit optimization and constraints (as in the previous code)

def profit_function_adjusted(x, cost_price, quantity, shipping_cost, discount):
    optimized_price = x[0]
    return -( (optimized_price - cost_price - discount) * quantity - shipping_cost)

def competitor_price_constraint(x, row):
    competitor_price = row['competitor_price']
    return competitor_price * 1.1 - x[0]

def elasticity_constraint(x, row):
    predicted_quantity = row['quantity'] * (1 - row['price_elasticity'] * ((x[0] - row['Predicted Price']) / row['Predicted Price']))
    return predicted_quantity - row['quantity'] * 0.9

def minimum_profit_constraint(x, row):
    cost_price = row['cost price']
    profit_margin = (x[0] - cost_price - row['discount']) / cost_price
    return profit_margin - 0.15

def shipping_cost_constraint(x, row):
    total_price_with_shipping = x[0] + row['shipping_cost']
    return row['Predicted Price'] * 1.2 - total_price_with_shipping

def segment_based_constraint(x, row):
    if row['segment_Corporate'] == 1:
        return x[0] - 1.2 * row['cost price']
    if row['segment_Home Office'] == 1:
        return x[0] - 1.15 * row['cost price']
    return 0

def category_based_constraint(x, row):
    if row['category_Technology'] == 1:
        return x[0] - 1.3 * row['cost price']
    elif row['category_Office Supplies'] == 1:
        return x[0] - 1.1 * row['cost price']
    return 0

# Function to optimize price with constraints
def optimize_price_with_constraints(row):
    cost_price = row['cost price']
    quantity = row['quantity']
    shipping_cost = row['shipping_cost']
    discount = row['discount']
    
    initial_guess = [row['Predicted Price']]
    
    upper_bound = max(cost_price * 1.5, cost_price + 10)
    bounds = [(cost_price, upper_bound)]

    constraints = [
        {'type': 'ineq', 'fun': competitor_price_constraint, 'args': (row,)},
        {'type': 'ineq', 'fun': elasticity_constraint, 'args': (row,)},
        {'type': 'ineq', 'fun': minimum_profit_constraint, 'args': (row,)},
        {'type': 'ineq', 'fun': shipping_cost_constraint, 'args': (row,)},
        {'type': 'ineq', 'fun': segment_based_constraint, 'args': (row,)},
        {'type': 'ineq', 'fun': category_based_constraint, 'args': (row,)}
    ]
    
    result = minimize(profit_function_adjusted, initial_guess, args=(cost_price, quantity, shipping_cost, discount), 
                      bounds=bounds, constraints=constraints)
    
    optimized_price = result.x[0]
    optimized_profit = -(result.fun)
    
    return {
        'Original Price': row['sales'],
        'Cost Price': cost_price,
        'Predicted Price': row['Predicted Price'],
        'Optimized Price': optimized_price,
        'Discount': discount,
        'Quantity': quantity,
        'Profit': optimized_profit
    }

# Apply optimization function to each row
optimized_results = df.apply(optimize_price_with_constraints, axis=1)

# Convert results to a DataFrame
optimized_df = pd.DataFrame(list(optimized_results))

# Format the DataFrame for better display
optimized_df.columns = ['Original Price', 'Cost Price', 'Predicted Price', 'Optimized Price', 'Discount', 'Quantity', 'Profit']

# Display the table in a formatted style
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Print the formatted table
print(optimized_df)

# If you're using a Jupyter notebook, display the dataframe using:
from IPython.display import display
display(optimized_df)
