import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, Tuple
import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 5px;}
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>select {
        border: 1px solid #0066cc;
        border-radius: 5px;
    }
    .highlight {font-size: 24px; color: #0066cc; font-weight: bold;}
    .metric-box {background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# 1. Claim Analysis Module
def analyze_claim(claim_data: Dict) -> Dict:
    """Extract features from claim data."""
    try:
        claim_amount = float(claim_data.get('claim_amount', 0))
        if claim_amount < 0:
            raise ValueError("Claim amount cannot be negative")
        customer_tenure = float(claim_data.get('customer_tenure', 0))
        if customer_tenure < 0:
            raise ValueError("Customer tenure cannot be negative")
        previous_claims = float(claim_data.get('previous_claims', 0))
        if previous_claims < 0:
            raise ValueError("Previous claims cannot be negative")
        features = {
            'claim_id': claim_data.get('claim_id', 'unknown'),
            'claim_amount': claim_amount,
            'claim_severity': {'low': 1, 'medium': 2, 'high': 3}.get(claim_data.get('severity', 'low').lower(), 1),
            'customer_tenure': customer_tenure,
            'claim_type': claim_data.get('claim_type', 'unknown'),
            'previous_claims': previous_claims,
        }
        return features
    except Exception as e:
        raise ValueError(f"Error analyzing claim: {str(e)}")

# 2. Cost Prediction Module
class CostPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['claim_amount', 'claim_severity', 'customer_tenure', 'previous_claims']

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the cost prediction model."""
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, features: Dict) -> Tuple[float, Tuple[float, float]]:
        """Predict litigation cost and settlement range."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        X = pd.DataFrame([{k: v for k, v in features.items() if k in self.feature_names}])
        X_scaled = self.scaler.transform(X[self.feature_names])
        litigation_cost = self.model.predict(X_scaled)[0]
        settlement_range = (litigation_cost * 0.8, litigation_cost * 1.2)
        return litigation_cost, settlement_range

# 3. Settlement Optimization Module
def optimize_settlement(
    claim_features: Dict,
    litigation_cost: float,
    settlement_range: Tuple[float, float],
    weights: Dict[str, float] = None
) -> Dict:
    """Optimize settlement amount based on costs, satisfaction, and profitability."""
    weights = weights or {'cost': 0.4, 'satisfaction': 0.3, 'profitability': 0.3}
    
    def satisfaction_score(amount: float, claim_amount: float) -> float:
        return min(1.0, max(0.0, amount / claim_amount))

    def profitability_score(amount: float, claim_amount: float) -> float:
        return 1.0 - min(1.0, amount / (claim_amount * 1.5))

    def objective_function(amount: float) -> float:
        cost_score = 1.0 - min(1.0, amount / litigation_cost)
        sat_score = satisfaction_score(amount, claim_features['claim_amount'])
        prof_score = profitability_score(amount, claim_features['claim_amount'])
        return (
            weights['cost'] * cost_score +
            weights['satisfaction'] * sat_score +
            weights['profitability'] * prof_score
        )

    amounts = np.linspace(settlement_range[0], settlement_range[1], 100)
    scores = [objective_function(amount) for amount in amounts]
    optimal_amount = amounts[np.argmax(scores)]
    risk_score = min(1.0, claim_features['claim_severity'] / 3.0 * (1 - claim_features['customer_tenure'] / 10.0))
    
    return {
        'optimal_amount': optimal_amount,
        'satisfaction_score': satisfaction_score(optimal_amount, claim_features['claim_amount']),
        'profitability_score': profitability_score(optimal_amount, claim_features['claim_amount']),
        'litigation_cost_avoided': litigation_cost - optimal_amount,
        'risk_score': risk_score,
        'sensitivity_data': {'amounts': amounts.tolist(), 'scores': scores}  # For line chart
    }

# 4. Recommendation Engine
def generate_recommendation(optimization_result: Dict, claim_features: Dict) -> Dict:
    """Generate settlement recommendation with cost-benefit analysis."""
    recommendation = {
        'claim_id': claim_features.get('claim_id', 'unknown'),
        'recommended_amount': optimization_result['optimal_amount'],
        'cost_benefit_analysis': {
            'litigation_cost_avoided': optimization_result['litigation_cost_avoided'],
            'satisfaction_impact': optimization_result['satisfaction_score'],
            'profitability_impact': optimization_result['profitability_score'],
            'litigation_risk': optimization_result['risk_score']
        },
        'confidence': min(0.95, 1.0 - optimization_result['risk_score']),
        'sensitivity_data': optimization_result['sensitivity_data']
    }
    return recommendation

# 5. Streamlit Dashboard
def run_dashboard():
    st.set_page_config(page_title="Claims Settlement System", layout="wide")
    st.title("Claims Settlement Optimization System")
    st.markdown("**Optimize insurance claim settlements with AI-driven insights**", unsafe_allow_html=True)

    # Input form in two columns
    st.header("Enter Claim Details")
    col1, col2 = st.columns(2)
    with col1:
        claim_id = st.text_input("Claim ID", "CLAIM_001")
        claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=10000.0, step=100.0)
        severity = st.selectbox("Claim Severity", ["Low", "Medium", "High"])
    with col2:
        customer_tenure = st.number_input("Customer Tenure (years)", min_value=0.0, value=5.0, step=0.5)
        previous_claims = st.number_input("Previous Claims", min_value=0.0, value=0.0, step=1.0)
        claim_type = st.text_input("Claim Type", "Property")

    claim_data = {
        'claim_id': claim_id,
        'claim_amount': claim_amount,
        'severity': severity,
        'customer_tenure': customer_tenure,
        'previous_claims': previous_claims,
        'claim_type': claim_type
    }

    if st.button("Analyze Claim", key="analyze"):
        try:
            # Analyze claim
            features = analyze_claim(claim_data)
            
            # Simulate trained model
            np.random.seed(42)
            X = pd.DataFrame({
                'claim_amount': np.random.uniform(1000, 50000, 1000),
                'claim_severity': np.random.randint(1, 4, 1000),
                'customer_tenure': np.random.uniform(0, 10, 1000),
                'previous_claims': np.random.randint(0, 5, 1000)
            })
            y = X['claim_amount'] * np.random.uniform(0.5, 1.5, 1000)
            predictor = CostPredictor()
            predictor.train(X, y)
            
            # Predict costs
            litigation_cost, settlement_range = predictor.predict(features)
            
            # Optimize settlement
            optimization_result = optimize_settlement(features, litigation_cost, settlement_range)
            
            # Generate recommendation
            recommendation = generate_recommendation(optimization_result, claim_data)
            
            # Display results
            st.header("Settlement Recommendation")
            st.markdown(f"<p class='highlight'>Recommended Settlement: ${recommendation['recommended_amount']:,.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"**Confidence**: {recommendation['confidence']*100:.1f}%", unsafe_allow_html=True)
            
            # Summary metrics in metric boxes
            st.subheader("Cost-Benefit Analysis")
            col1, col2, col3, col4 = st.columns(4)
            metrics = recommendation['cost_benefit_analysis']
            with col1:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Litigation Cost Avoided", f"${metrics['litigation_cost_avoided']:,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Satisfaction Impact", f"{metrics['satisfaction_impact']*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Profitability Impact", f"{metrics['profitability_impact']*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            with col4:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Litigation Risk", f"{metrics['litigation_risk']*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Bar chart
            st.subheader("Cost-Benefit Visualization")
            st.bar_chart({
                'Litigation Cost Avoided': metrics['litigation_cost_avoided'],
                'Satisfaction Impact': metrics['satisfaction_impact'] * 100,
                'Profitability Impact': metrics['profitability_impact'] * 100,
                'Litigation Risk': metrics['litigation_risk'] * 100
            })
            
            # Sensitivity analysis
            with st.expander("Sensitivity Analysis"):
                st.write("How the settlement amount varies with optimization weights")
                sensitivity_df = pd.DataFrame({
                    'Settlement Amount ($)': recommendation['sensitivity_data']['amounts'],
                    'Optimization Score': recommendation['sensitivity_data']['scores']
                })
                st.line_chart(sensitivity_df.set_index('Settlement Amount ($)'))
            
            # Detailed JSON output
            with st.expander("Detailed Recommendation (JSON)"):
                st.json(recommendation)
            
            st.success("Claim analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error processing claim: {str(e)}")

if __name__ == "__main__":
    run_dashboard()