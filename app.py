import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

# ====================================================================
# Flask Application Setup
# ====================================================================

app = Flask(__name__, static_folder="static", template_folder="templates")

# The list of features and their required order for the model
FEATURES = ['unit_price', 'comp_1', 'comp_2', 'comp_3', 'holiday', 'weekend', 'month']

# ====================================================================
# Machine Learning Model Training
# ====================================================================

# This function trains the model when the application starts
def train_model():
    """
    Loads data, trains a RandomForestRegressor model, and evaluates it.
    Returns the trained model and the original DataFrame for feature scaling.
    """
    try:
        # Assumes the dataset is in the same directory as this file
        df = pd.read_csv("retail_price.csv")
    except FileNotFoundError:
        print("Error: 'retail_price.csv' not found. Please ensure the file is in the same directory.")
        return None, None

    # Define the features (X) and the target variable (y)
    # The order of columns in the DataFrame must match FEATURES list
    X = df[FEATURES]
    y = df['qty']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Model trained successfully. Root Mean Squared Error (RMSE): {rmse:.2f}")

    return model, df

# Train the model once when the app starts
model, df_original = train_model()

# ====================================================================
# Helper Function for Optimal Price Calculation
# ====================================================================

def find_optimal_price(base_features):
    """
    Finds the optimal unit price that maximizes predicted revenue.
    
    Args:
        base_features (dict): A dictionary of the user's input features
                              (comp_1, comp_2, etc.), excluding unit_price.

    Returns:
        A dictionary with the optimal price, demand, and revenue.
    """
    if df_original is None:
        return None

    # Generate a range of prices to test, from min to max of the dataset
    price_range = np.linspace(df_original['unit_price'].min(), df_original['unit_price'].max(), 100)
    
    # Create a list of dictionaries, one for each price point to test
    prediction_data = []
    for p in price_range:
        temp_data = base_features.copy()
        temp_data['unit_price'] = p
        prediction_data.append(temp_data)

    # Create the DataFrame for prediction, ensuring columns are in the correct order
    df_temp = pd.DataFrame(prediction_data, columns=FEATURES)

    # Predict demand for all prices in the range
    predicted_demand_array = model.predict(df_temp)
    predicted_revenue_array = price_range * predicted_demand_array

    # Find the maximum revenue and its corresponding price and demand
    max_revenue_index = np.argmax(predicted_revenue_array)
    max_revenue = predicted_revenue_array[max_revenue_index]
    optimal_price = price_range[max_revenue_index]
    optimal_demand = predicted_demand_array[max_revenue_index]

    return {
        'optimal_price': round(optimal_price, 2),
        'optimal_demand': round(optimal_demand, 2),
        'max_revenue': round(max_revenue, 2)
    }

# ====================================================================
# Flask Routes
# ====================================================================

@app.route("/")
def home():
    """Renders the main HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles form submission, predicts demand and optimal price, and returns JSON data.
    """
    if model is None:
        return jsonify({"error": "Model not trained. Check server logs."}), 500

    try:
        # Get user input from the form
        user_input_data = {
            "unit_price": float(request.form['unit_price']),
            "comp_1": float(request.form['comp_1']),
            "comp_2": float(request.form['comp_2']),
            "comp_3": float(request.form['comp_3']),
            "holiday": int(request.form['holiday']),
            "weekend": int(request.form['weekend']),
            "month": int(request.form['month'])
        }

        # Create the DataFrame for prediction, explicitly setting the column order
        df_input = pd.DataFrame([user_input_data], columns=FEATURES)
        
        # Predict demand and revenue for the user's chosen price
        user_demand = model.predict(df_input)[0]
        user_revenue = user_input_data["unit_price"] * user_demand

        # Find the overall optimal price for revenue maximization
        base_features = {k: v for k, v in user_input_data.items() if k != 'unit_price'}
        optimal_results = find_optimal_price(base_features)

        # Generate data for the plots (Demand Curve and Revenue Curve)
        price_range = np.linspace(df_original['unit_price'].min(), df_original['unit_price'].max(), 100)
        
        # Build the list of dictionaries for the plot data, ensuring correct order
        plot_data_list = []
        for p in price_range:
            temp_data = base_features.copy()
            temp_data['unit_price'] = p
            plot_data_list.append(temp_data)
        
        df_plot = pd.DataFrame(plot_data_list, columns=FEATURES)

        plot_demand = model.predict(df_plot).tolist()
        plot_revenue = (price_range * np.array(plot_demand)).tolist()
        
        # Prepare the final JSON response
        response = {
            "success": True,
            "user_prediction": {
                "demand": round(user_demand, 2),
                "revenue": round(user_revenue, 2),
            },
            "optimal_prediction": optimal_results,
            "plot_data": {
                "prices": price_range.tolist(),
                "demand": plot_demand,
                "revenue": plot_revenue,
            }
        }
        
        return jsonify(response)

    except Exception as e:
        # Return a user-friendly error message
        print(f"Error processing request: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    # Start the Flask development server
    app.run(debug=True)