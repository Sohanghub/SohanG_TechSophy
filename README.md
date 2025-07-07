# Claims Settlement AI

A web-based machine learning system that recommends optimal insurance claim settlement amounts. Built using Streamlit, scikit-learn, and Plotly, this tool provides an interactive dashboard with intelligent cost prediction, risk analysis, and performance insights.

---

## Features

- Machine learning-based cost prediction using Random Forest
- Settlement optimization balancing cost, satisfaction, and profitability
- Sensitivity analysis to understand settlement impact
- Clean, modern dashboard UI with interactive visualizations
- Dynamic metrics for decision support

---

## Project Structure

```
claims-settlement-ai/
│
├── app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## Installation

Ensure you have **Python 3.8 or higher** installed.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Required Libraries

Create a `requirements.txt` with the following:

```
streamlit
scikit-learn
pandas
numpy
plotly
```

Then install:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the following command to launch the app:

```bash
streamlit run app.py
```

Then open the URL `http://localhost:8501` in your browser.

---

## How It Works

1. The user inputs claim information including amount, severity, and customer history.
2. A machine learning model predicts the potential litigation cost.
3. An optimization algorithm suggests a settlement amount to minimize risk and maximize satisfaction.
4. Visual dashboards display key metrics and sensitivity insights.
5. A recommendation is generated with a confidence score and cost-benefit breakdown.

---

## Tech Stack

| Component     | Technology         |
|---------------|--------------------|
| UI Framework  | Streamlit          |
| ML Algorithm  | RandomForestRegressor (scikit-learn) |
| Data Handling | Pandas, NumPy      |
| Visualization | Plotly             |
| Language      | Python             |

---

## Future Enhancements

- Integrate real-world claim data
- Model persistence and loading from storage
- Claim history and audit log
- Deployment via Streamlit Cloud or cloud infrastructure
- Improved user authentication and access control
