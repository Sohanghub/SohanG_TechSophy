import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, Tuple
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Claims Settlement System", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏛️"
)

# Modern CSS with glassmorphism and contemporary design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        min-height: 100vh;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Modern button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input field styling */
    .stTextInput > div > input, 
    .stNumberInput > div > input, 
    .stSelectbox > div > select {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 0.8rem;
        color: #ffffff;
        font-size: 1rem;
        backdrop-filter: blur(5px);
    }
    
    .stTextInput > div > input:focus, 
    .stNumberInput > div > input:focus, 
    .stSelectbox > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.1));
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Recommendation highlight */
    .recommendation-highlight {
        background: linear-gradient(45deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .recommendation-amount {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Form styling */
    .form-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        border-radius: 12px;
        padding: 1rem;
        font-weight: 500;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0 0 12px 12px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Label styling */
    .stTextInput > label, 
    .stNumberInput > label, 
    .stSelectbox > label {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
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
        'sensitivity_data': {'amounts': amounts.tolist(), 'scores': scores}
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

# 5. Enhanced Visualization Functions
def create_metrics_chart(metrics):
    """Create a modern donut chart for metrics."""
    fig = go.Figure(data=[go.Pie(
        labels=['Satisfaction', 'Profitability', 'Risk Reduction'],
        values=[
            metrics['satisfaction_impact'] * 100,
            metrics['profitability_impact'] * 100,
            (1 - metrics['litigation_risk']) * 100
        ],
        hole=0.6,
        marker=dict(colors=['#667eea', '#764ba2', '#48bb78'])
    )])
    
    fig.update_layout(
        title="Performance Metrics",
        title_font_size=20,
        title_font_color='white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_sensitivity_chart(sensitivity_data):
    """Create a modern sensitivity analysis chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sensitivity_data['amounts'],
        y=sensitivity_data['scores'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)',
        name='Optimization Score'
    ))
    
    # Add optimal point
    max_idx = np.argmax(sensitivity_data['scores'])
    fig.add_trace(go.Scatter(
        x=[sensitivity_data['amounts'][max_idx]],
        y=[sensitivity_data['scores'][max_idx]],
        mode='markers',
        marker=dict(color='#ffffff', size=12, line=dict(color='#667eea', width=2)),
        name='Optimal Point'
    ))
    
    fig.update_layout(
        title="Settlement Optimization Curve",
        xaxis_title="Settlement Amount ($)",
        yaxis_title="Optimization Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_size=18,
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

# 6. Modern Streamlit Dashboard
def run_dashboard():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🏛️ Claims Settlement AI</h1>
            <p class="subtitle">Intelligent settlement optimization powered by machine learning</p>
        </div>
    """, unsafe_allow_html=True)

    # Input form
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📋 Claim Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        claim_id = st.text_input("🆔 Claim ID", "CLAIM_001")
        claim_amount = st.number_input("💰 Claim Amount ($)", min_value=0.0, value=25000.0, step=100.0)
        severity = st.selectbox("⚠️ Claim Severity", ["Low", "Medium", "High"], index=1)
    
    with col2:
        customer_tenure = st.number_input("👤 Customer Tenure (years)", min_value=0.0, value=5.0, step=0.5)
        previous_claims = st.number_input("📊 Previous Claims", min_value=0.0, value=1.0, step=1.0)
        claim_type = st.text_input("🏷️ Claim Type", "Property Damage")

    st.markdown('</div>', unsafe_allow_html=True)

    claim_data = {
        'claim_id': claim_id,
        'claim_amount': claim_amount,
        'severity': severity,
        'customer_tenure': customer_tenure,
        'previous_claims': previous_claims,
        'claim_type': claim_type
    }

    # Analyze button
    if st.button("🚀 Analyze & Optimize Settlement", key="analyze"):
        with st.spinner("🔄 Processing claim data..."):
            try:
                # Analyze claim
                features = analyze_claim(claim_data)
                
                # Simulate trained model
                np.random.seed(42)
                X = pd.DataFrame({
                    'claim_amount': np.random.uniform(1000, 50000, 1000),
                    'claim_severity': np.random.randint(1, 4, 1000),
                    'customer_tenure': np.random.uniform(0, 15, 1000),
                    'previous_claims': np.random.randint(0, 8, 1000)
                })
                y = X['claim_amount'] * np.random.uniform(0.6, 1.8, 1000)
                predictor = CostPredictor()
                predictor.train(X, y)
                
                # Predict costs
                litigation_cost, settlement_range = predictor.predict(features)
                
                # Optimize settlement
                optimization_result = optimize_settlement(features, litigation_cost, settlement_range)
                
                # Generate recommendation
                recommendation = generate_recommendation(optimization_result, claim_data)
                
                # Display results
                st.markdown('<div class="recommendation-highlight">', unsafe_allow_html=True)
                st.markdown(f'<div class="recommendation-amount">${recommendation["recommended_amount"]:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size: 1.2rem; margin-bottom: 1rem;">Recommended Settlement Amount</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-badge">✅ {recommendation["confidence"]*100:.1f}% Confidence</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics cards
                st.markdown('<h2 class="section-header">📈 Performance Metrics</h2>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                metrics = recommendation['cost_benefit_analysis']
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">${metrics['litigation_cost_avoided']:,.0f}</div>
                            <div class="metric-label">Cost Avoided</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{metrics['satisfaction_impact']*100:.1f}%</div>
                            <div class="metric-label">Satisfaction</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{metrics['profitability_impact']*100:.1f}%</div>
                            <div class="metric-label">Profitability</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{(1-metrics['litigation_risk'])*100:.1f}%</div>
                            <div class="metric-label">Risk Reduction</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown('<h2 class="section-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    metrics_chart = create_metrics_chart(metrics)
                    st.plotly_chart(metrics_chart, use_container_width=True)
                
                with col2:
                    sensitivity_chart = create_sensitivity_chart(recommendation['sensitivity_data'])
                    st.plotly_chart(sensitivity_chart, use_container_width=True)
                
                # Expandable sections
                with st.expander("🔍 Detailed Analysis"):
                    st.json(recommendation)
                
                with st.expander("⚙️ Model Parameters"):
                    st.write("**Feature Importance:**")
                    st.write("- Claim Amount: High impact on settlement calculation")
                    st.write("- Customer Tenure: Affects risk assessment")
                    st.write("- Claim Severity: Influences litigation probability")
                    st.write("- Previous Claims: Historical risk indicator")
                
                st.success("✅ Analysis completed successfully! Settlement recommendation generated.")
                
            except Exception as e:
                st.error(f"❌ Error processing claim: {str(e)}")

if __name__ == "__main__":
    run_dashboard()