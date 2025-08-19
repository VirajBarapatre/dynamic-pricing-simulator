import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)

# ---- Train model once ----
df = pd.read_csv(r"C:\Users\viraj\Dynamic Pricing Simulation\retail_price.csv")
features = ['unit_price', 'comp_1', 'comp_2', 'comp_3', 'holiday', 'weekend', 'month']
X = df[features]
y = df['qty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Model trained. RMSE:", rmse)

# Create folder for plots
PLOT_FOLDER = "static/plots"
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "unit_price": float(request.form['unit_price']),
            "comp_1": float(request.form['comp_1']),
            "comp_2": float(request.form['comp_2']),
            "comp_3": float(request.form['comp_3']),
            "holiday": int(request.form['holiday']),
            "weekend": int(request.form['weekend']),
            "month": int(request.form['month'])
        }

        df_input = pd.DataFrame([data])
        demand = model.predict(df_input)[0]
        revenue = data["unit_price"] * demand

        # Generate demand vs price curve
        price_range = np.linspace(df['unit_price'].min(), df['unit_price'].max(), 50)
        demand_curve = []
        for p in price_range:
            temp = df_input.copy()
            temp['unit_price'] = p
            demand_curve.append(model.predict(temp)[0])

        # Plot the curve
        fig, ax = plt.subplots()
        ax.plot(price_range, demand_curve, label="Predicted Demand", color='blue')
        ax.axvline(data['unit_price'], color='red', linestyle='--', label='Current Price')
        ax.set_xlabel("Unit Price")
        ax.set_ylabel("Predicted Demand")
        ax.set_title("Demand vs. Price Curve")
        ax.legend()

        plot_filename = f"{uuid.uuid4().hex}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        return render_template("index.html", prediction=True,
                               demand=round(demand, 2),
                               revenue=round(revenue, 2),
                               plot_file=plot_filename)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
