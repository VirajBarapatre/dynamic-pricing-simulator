# generate_extended_dataset.py
import pandas as pd
import numpy as np

np.random.seed(42)

# Catalog with 20 products per category
catalog = {
    "Smartphones & Tablets": {
        "products": [
            "iPhone 15 Pro", "Galaxy S23 Ultra", "Google Pixel 8 Pro", "OnePlus 12",
            "Xiaomi 14 Ultra", "iPhone SE 3", "Galaxy Z Flip 5", "Samsung Z Fold 5",
            "iPad Air", "iPad Mini 6", "Surface Pro 9", "Galaxy Tab S9",
            "Lenovo Tab P12 Pro", "Asus ROG Phone 8", "Motorola Razr+",
            "Nokia G400", "Pixel 7a", "Galaxy A54 5G", "Google Pixel Fold",
            "Galaxy Tab A9"
        ],
        "price_min": 300, "price_max": 1600
    },
    "Laptops & Computers": {
        "products": [
            "MacBook Pro M3", "Dell XPS 15", "Lenovo ThinkPad X1 Carbon", "HP Spectre x360",
            "Asus Zenbook 14", "Surface Laptop 5", "Razer Blade 16", "Alienware m18",
            "Acer Swift Go 14", "LG Gram 17", "Dell Inspiron 16", "HP Envy x360",
            "Lenovo Yoga 7i", "Asus Vivobook 15", "Acer Aspire 5", "Gateway 15.6 Laptop",
            "Chromebook Duet 5", "MSI Thin GF63", "Dell G15 Gaming", "Lenovo Legion 5i"
        ],
        "price_min": 500, "price_max": 2800
    },
    "Wearables & Gadgets": {
        "products": [
            "Apple Watch Ultra 2", "Galaxy Watch 6", "Google Pixel Watch 2", "Fitbit Sense 2",
            "Garmin Forerunner 965", "Oura Ring Gen3", "Whoop 4.0", "Withings ScanWatch 2",
            "Amazfit GTR 4", "Suunto Vertical", "Polar Vantage V3", "Coros Pace 3",
            "Garmin Fenix 7 Pro", "Samsung Buds2 Pro", "Apple AirPods Pro 2",
            "Bose QuietComfort", "JBL Live 670NC", "Sony WH-1000XM5",
            "Anker Soundcore Space", "Beats Studio Pro"
        ],
        "price_min": 100, "price_max": 600
    },
    "Home & Kitchen": {
        "products": [
            "Dyson V15 Detect", "Shark Stratos Cordless", "iRobot Roomba j7+",
            "Roborock S8 Pro Ultra", "Eufy RoboVac 11S", "Keurig K-Elite",
            "Breville Barista Express", "Nespresso VertuoPlus", "Vitamix Ascent Series",
            "Ninja Foodi 8-in-1", "Instant Pot Duo Crisp", "Cuisinart Air Fryer",
            "KitchenAid Artisan Mixer", "Zojirushi Rice Cooker", "Philips Pasta Maker",
            "Cuisinart Coffee Center", "Hamilton Beach FlexBrew", "NutriBullet Pro",
            "Oster Versa Blender", "Ninja Professional Blender"
        ],
        "price_min": 50, "price_max": 1200
    }
}

category_sensitivity_anchor = {
    "Smartphones & Tablets": 1.3,
    "Laptops & Computers":   1.6,
    "Wearables & Gadgets":   2.2,
    "Home & Kitchen":        1.9
}

rows = []
SAMPLES_PER_MONTH_PER_PRODUCT = 12

for category, spec in catalog.items():
    pmin, pmax = spec["price_min"], spec["price_max"]
    for product in spec["products"]:
        # 🎯 Each product gets its own base price + elasticity
        base_price = np.random.uniform(pmin*1.1, pmax*0.9)
        elasticity = np.random.uniform(0.8, 2.5)  # product-specific
        base_demand_level = np.random.randint(1000, 2500)  # product-specific

        for month in range(1, 13):
            seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (month - 1) / 12.0)

            for _ in range(SAMPLES_PER_MONTH_PER_PRODUCT):
                unit_price = np.clip(
                    np.random.normal(base_price, base_price * 0.15),
                    pmin, pmax
                )
                promotion = np.random.choice([0, 1], p=[0.7, 0.3])
                holiday = np.random.choice([0, 1], p=[0.88, 0.12])
                weekend = np.random.choice([0, 1], p=[0.65, 0.35])

                comp_1 = unit_price * np.random.uniform(0.85, 1.15)
                comp_2 = unit_price * np.random.uniform(0.85, 1.15)
                comp_3 = unit_price * np.random.uniform(0.85, 1.15)

                demand = (
                    base_demand_level * seasonal
                    - elasticity * unit_price
                    + (comp_1+comp_2+comp_3)*0.03
                    + promotion*250 + holiday*150 + weekend*70
                    + np.random.normal(0, 50)
                )
                qty = max(0, int(round(demand)))

                rows.append([
                    product, category, promotion, unit_price,
                    comp_1, comp_2, comp_3, holiday, weekend, month, qty
                ])

df = pd.DataFrame(rows, columns=[
    "product_name","product_category","promotion","unit_price",
    "comp_1","comp_2","comp_3","holiday","weekend","month","qty"
])
df.to_csv("extended_retail_data.csv", index=False)
print(f"✅ Generated dataset with {len(df):,} rows and {df['product_name'].nunique()} products.")
