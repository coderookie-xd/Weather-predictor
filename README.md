🌤️ NYC Weather Prediction using Random Forest
📌 Project Overview

This project predicts average daily temperature (tavg) in New York City using historical weather data.
We use a Random Forest Regressor to learn patterns from features like max/min temperature, heating/cooling degree days, and departure from normal.

⚙️ Features Used

tmax → Maximum temperature of the day

tmin → Minimum temperature of the day

departure → Departure from normal temperature

HDD → Heating Degree Days

CDD → Cooling Degree Days

Target Variable:

tavg → Average temperature (predicted value)

🛠️ Tech Stack

Python 🐍

pandas, numpy → Data handling

scikit-learn → Machine Learning (Random Forest, train/test split, evaluation)

matplotlib, seaborn → Visualization (optional)

📂 Project Workflow

Import Libraries
Load required libraries for ML and data analysis.

Load Dataset

df = pd.read_csv("nyc_temperature.csv")


Feature Selection

X = df[["tmax", "tmin", "departure", "HDD", "CDD"]]
y = df["tavg"]


Train-Test Split
Dataset is split:

80% → Training

20% → Testing

Model Training

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


Evaluation

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

📊 Results

MSE: Very low (close to 0)

R² Score: Very high (close to 1)
→ The model performs strongly, showing that Random Forest is effective for weather prediction.

🚀 How to Run

Clone the repo

git clone https://github.com/your-username/nyc-weather-prediction.git
cd nyc-weather-prediction


Install dependencies

pip install pandas numpy scikit-learn matplotlib seaborn


Run the Jupyter Notebook or Python script.

📌 Future Improvements

Add more features (humidity, wind, precipitation).

Try other models (XGBoost, Linear Regression, Neural Networks).

Build a simple visualization dashboard for results.

🔥 Built with Random Forest & NYC Weather Data
