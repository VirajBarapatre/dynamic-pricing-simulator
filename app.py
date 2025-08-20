import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import json

# ====================================================================
# Flask Application Setup
# ====================================================================

# The template and static folders must be in the same directory as this file
app = Flask(__name__, static_folder="static", template_folder="templates")

# The list of features and their required order for the model
FEATURES = ['product_name', 'unit_price', 'comp_1', 'comp_2', 'comp_3', 'holiday', 'weekend', 'month']

# ====================================================================
# Machine Learning Model Training with Pipeline
# ====================================================================

# This function generates a synthetic dataset for demonstration.
# In a real project, you would replace this with loading your own data.
def load_and_prepare_data():
    """Generates a synthetic dataset with product names."""
    product_names = [
        "Smartphone Pro 256GB Black",
        "Smartphone Lite 64GB Blue",
        "Laptop XPS 15 inch",
        "Laptop ThinkPad E14",
        "Smart Watch Series 7 GPS",
        "Smart Watch GT Pro",
        "Gaming Mouse G502 HERO",
        "Wireless Keyboard K400",
    ]
    np.random.seed(42)
    data = {
        'product_name': np.random.choice(product_names, 1000),
        'unit_price': np.random.uniform(100, 1000, 1000),
        'comp_1': np.random.uniform(90, 1100, 1000),
        'comp_2': np.random.uniform(90, 1100, 1000),
        'comp_3': np.random.uniform(90, 1100, 1000),
        'holiday': np.random.randint(0, 2, 1000),
        'weekend': np.random.randint(0, 2, 1000),
        'month': np.random.randint(1, 13, 1000)
    }
    df = pd.DataFrame(data)
    df['qty'] = (
        1500 - 
        df['unit_price'] * 0.8 + 
        df['comp_1'] * 0.1 + 
        df['comp_2'] * 0.05 + 
        df['holiday'] * 50 + 
        df['weekend'] * 20 + 
        np.random.normal(0, 30, 1000)
    )
    df['qty'] = df['qty'].apply(lambda x: max(0, x)).astype(int)
    
    return df

# The function that trains the model pipeline
def train_model():
    """
    Loads data, sets up a preprocessing pipeline, and trains the model.
    """
    df = load_and_prepare_data()
    
    numerical_features = ['unit_price', 'comp_1', 'comp_2', 'comp_3', 'holiday', 'weekend', 'month']
    text_feature = 'product_name'
    target_variable = 'qty'

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_vectorizer', TfidfVectorizer(max_features=100), text_feature)
        ],
        remainder='passthrough'  # Keep the numerical features as they are
    )

    # Create a pipeline that combines preprocessing and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training model pipeline...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Model trained successfully. Root Mean Squared Error (RMSE): {rmse:.2f}")

    return model_pipeline, df

# Train the model once when the app starts
model, df_original = train_model()

# ====================================================================
# Helper Function for Optimal Price Calculation
# ====================================================================

def find_optimal_price(base_features, unit_cost):
    """
    Finds the optimal unit price that maximizes predicted profit.
    """
    if df_original is None:
        return None

    price_range = np.linspace(df_original['unit_price'].min(), df_original['unit_price'].max(), 100)
    
    # Create DataFrame for prediction, ensuring all features are present
    prediction_data = []
    for p in price_range:
        temp_data = base_features.copy()
        temp_data['unit_price'] = p
        prediction_data.append(temp_data)

    df_temp = pd.DataFrame(prediction_data, columns=base_features.keys())

    predicted_demand_array = model.predict(df_temp)
    predicted_revenue_array = price_range * predicted_demand_array
    predicted_profit_array = predicted_revenue_array - (unit_cost * predicted_demand_array)

    max_profit_index = np.argmax(predicted_profit_array)
    max_profit = predicted_profit_array[max_profit_index]
    optimal_price = price_range[max_profit_index]
    optimal_demand = predicted_demand_array[max_profit_index]

    return {
        'optimal_price': round(optimal_price, 2),
        'optimal_demand': round(optimal_demand, 2),
        'max_profit': round(max_profit, 2)
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
        return jsonify({"success": False, "error": "Model not trained. Check server logs."}), 500

    try:
        # Get user input from the form, including the new product_name field
        user_input_data = {
            "product_name": request.form['product_name'],
            "unit_cost": float(request.form['unit_cost']),
            "unit_price": float(request.form['unit_price']),
            "comp_1": float(request.form['comp_1']),
            "comp_2": float(request.form['comp_2']),
            "comp_3": float(request.form['comp_3']),
            "holiday": int(request.form['holiday']),
            "weekend": int(request.form['weekend']),
            "month": int(request.form['month'])
        }

        # Create the DataFrame for prediction, explicitly setting the column order
        df_input_features = {k: v for k, v in user_input_data.items() if k != 'unit_cost'}
        df_input = pd.DataFrame([df_input_features], columns=FEATURES)
        
        # Predict demand, revenue, and profit for the user's chosen price
        user_demand = model.predict(df_input)[0]
        user_revenue = user_input_data["unit_price"] * user_demand
        user_profit = user_revenue - (user_input_data["unit_cost"] * user_demand)

        # Find the overall optimal price for profit maximization
        base_features = {k: v for k, v in user_input_data.items() if k not in ['unit_cost', 'unit_price']}
        optimal_results = find_optimal_price(base_features, user_input_data['unit_cost'])

        # Generate data for the plots (Demand Curve and Profit Curve)
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
        plot_profit = (np.array(plot_revenue) - (user_input_data['unit_cost'] * np.array(plot_demand))).tolist()

        # Prepare the final JSON response
        response = {
            "success": True,
            "user_prediction": {
                "demand": round(user_demand, 2),
                "revenue": round(user_revenue, 2),
                "profit": round(user_profit, 2)
            },
            "optimal_prediction": optimal_results,
            "plot_data": {
                "prices": price_range.tolist(),
                "demand": plot_demand,
                "profit": plot_profit,
            }
        }
        
        return jsonify(response)

    except KeyError as e:
        # Catch specific KeyErrors for missing form fields
        error_message = f"Missing form field: {e}. Please ensure all form fields are filled correctly."
        print(f"Error processing request: {error_message}")
        return jsonify({"success": False, "error": error_message}), 400
    except ValueError as e:
        # Catch ValueErrors for invalid numeric input
        error_message = f"Invalid input: {e}. Please ensure all numeric fields are valid."
        print(f"Error processing request: {error_message}")
        return jsonify({"success": False, "error": error_message}), 400
    except Exception as e:
        # Catch any other unexpected errors
        error_message = f"An unexpected error occurred: {e}"
        print(f"Error processing request: {error_message}")
        return jsonify({"success": False, "error": error_message}), 400

if __name__ == "__main__":
    # Start the Flask development server
    app.run(debug=True)