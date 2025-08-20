# AI Dynamic Pricing Simulator

An **AI-powered simulator** that predicts **demand, revenue, and profit** for products across multiple categories and recommends the **optimal price** using Machine Learning.  

🌐 **Live Demo**: [View on Render](https://dynamic-pricing-simulator.onrender.com/)  Render may take ~30s cold start

---

## 🚀 Features
- 📊 Predicts **demand, revenue, profit** for user-input price
- 💡 Recommends **optimal product price** using ML models
- 🎨 Interactive **web interface** with Flask + Tailwind CSS
- 📈 Visualizes **Demand & Profit curves** with Chart.js
- 🛒 Dataset generator simulates **real-world market data** (20+ products per category)
- ⚡ Trained with **Random Forest** for accuracy & fast predictions  

---

## 📂 Project Structure
ai-dynamic-pricing-simulator/
├── app.py # Main Flask app
├── extended_retail_data.py # Generates synthetic dataset
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Ignore cache & venv files
│
├── data/
│ └── extended_retail_data.csv # Generated dataset (optional, can regenerate)
│
├── templates/
│ └── index.html # Frontend page

---
## ⚡ How to Run Locally
1. **Clone Repo**
```bash
git clone https://github.com/yourusername/ai-dynamic-pricing-simulator.git
cd ai-dynamic-pricing-simulator
```


**Install Dependencies**
```bash
pip install -r requirements.txt
```

**Generate Dataset**
```bash
python dataset_generator.py
```

**Run App**
```bash
python app.py
```
App will run at 👉 http://127.0.0.1:5000

📸 Screenshots
**Web Interface**
<img width="500" height="1000" alt="dynamic-pricing-simulator onrender com_" src="https://github.com/user-attachments/assets/5ce9ea30-17cb-4e7a-abe5-9e91508a24d5" />

🛠️ Tech Stack
Backend: Flask (Python)
ML: scikit-learn, pandas, numpy
Frontend: Tailwind CSS, Chart.js
Deployment: Render

📈 Future Improvements
🔑 User authentication for business accounts
📊 Upload & train on real-world pricing datasets
🤖 Explore Reinforcement Learning for dynamic pricing updates
☁️ Deploy on AWS/GCP/Azure for scalability

👨‍💻 Author
Developed by Viraj Barapatre ✨
📌 GitHub: https://github.com/VirajBarapatre
