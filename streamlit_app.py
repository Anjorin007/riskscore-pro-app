import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import cohere
import io
from fpdf import FPDF
import unicodedata
import os
import numpy as np

# -------------------------------
# OPTIMISATIONS DE PERFORMANCE
# -------------------------------
# Cache pour éviter de recharger le modèle à chaque fois
@st.cache_resource
def load_model():
    return joblib.load("../models/xgb_model.pkl")

# Cache pour l'explainer SHAP (très coûteux à créer)
@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

# Cache pour les prédictions (évite les recalculs inutiles)
@st.cache_data
def predict_client(age, income, dependents, open_credit, real_estate, debt_ratio, revolving, late_30, late_60, late_90):
    client_df = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': revolving / 100,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': income,
        'NumberOfOpenCreditLinesAndLoans': open_credit,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60,
        'NumberOfDependents': dependents
    }])
    
    model = load_model()
    proba = model.predict_proba(client_df)[0][1]
    classe = model.predict(client_df)[0]
    
    return client_df, proba, classe

# Cache pour les calculs SHAP (très coûteux)
@st.cache_data
def compute_shap_values(age, income, dependents, open_credit, real_estate, debt_ratio, revolving, late_30, late_60, late_90):
    client_df, _, _ = predict_client(age, income, dependents, open_credit, real_estate, debt_ratio, revolving, late_30, late_60, late_90)
    model = load_model()
    explainer = get_shap_explainer(model)
    shap_values = explainer(client_df)
    
    # Retourner seulement les données nécessaires
    shap_df = pd.DataFrame({
        'feature': client_df.columns,
        'shap_value': shap_values[0].values,
        'value': client_df.iloc[0].values
    }).sort_values(by='shap_value', key=abs, ascending=False)
    
    return shap_values[0], shap_df

# Fonction pour générer automatiquement le rapport IA basé sur les données actuelles
def generate_auto_ai_report(age, income, dependents, open_credit, real_estate, debt_ratio, revolving, late_30, late_60, late_90, proba, classe, shap_df, lang):
    # Rapport automatique basé sur les données
    risk_level = "élevé" if classe == 1 else "faible"
    income_fcfa = f"{income:,} FCFA"
    
    top_factors = shap_df.head(3)
    factors_text = ""
    for _, row in top_factors.iterrows():
        impact = "augmente" if row['shap_value'] > 0 else "diminue"
        feature_name = row['feature'].replace('RevolvingUtilizationOfUnsecuredLines', 'Crédit renouvelable')
        feature_name = feature_name.replace('NumberOfTime30-59DaysPastDueNotWorse', 'Retards 30-59j')
        feature_name = feature_name.replace('MonthlyIncome', 'Revenus mensuels')
        feature_name = feature_name.replace('age', 'Âge')
        factors_text += f"• {feature_name} {impact} le risque\n"
    
    if lang == "fr":
        report = f"""📊 ANALYSE AUTOMATIQUE DU PROFIL CLIENT

🎯 Risque évalué : {risk_level.upper()} ({proba:.1%} de probabilité de défaut)

👤 PROFIL CLIENT :
• Âge : {age} ans
• Revenus : {income_fcfa}
• Personnes à charge : {dependents}
• Ratio d'endettement : {debt_ratio}
• Utilisation crédit renouvelable : {revolving}%

📈 FACTEURS D'IMPACT PRINCIPAUX :
{factors_text}
🏦 RECOMMANDATION :
{'⚠️ Surveillance renforcée nécessaire. Conditions strictes recommandées.' if classe == 1 else '✅ Profil acceptable. Client éligible au crédit standard.'}"""
    else:
        risk_level_en = "high" if classe == 1 else "low" 
        report = f"""📊 AUTOMATIC CLIENT PROFILE ANALYSIS

🎯 Risk assessed: {risk_level_en.upper()} ({proba:.1%} probability of default)

👤 CLIENT PROFILE:
• Age: {age} years
• Income: {income_fcfa}
• Dependents: {dependents}
• Debt ratio: {debt_ratio}
• Revolving credit utilization: {revolving}%

📈 MAIN IMPACT FACTORS:
{factors_text}
🏦 RECOMMENDATION:
{'⚠️ Enhanced monitoring required. Strict conditions recommended.' if classe == 1 else '✅ Acceptable profile. Client eligible for standard credit.'}"""
    
    return report

# -------------------------------
# Fonction pour retirer les accents (pour PDF)
# -------------------------------
def remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

# -------------------------------
# Dictionnaire de traduction (modifié pour FCFA)
# -------------------------------
T = {
    "fr": {
        "page_title": "RiskScore Pro | Analyse de Crédit",
        "about": "Application professionnelle d'analyse du risque crédit basée sur l'IA.",
        "header_title": "RiskScore Pro - Évaluation du Risque Crédit",
        "header_subtitle": "Plateforme d'analyse prédictive fondée sur XGBoost, SHAP & IA générative",
        "profile_client": "Profil Client",
        "age": "Âge",
        "income": "Revenu mensuel (FCFA)",
        "dependents": "Personnes à charge",
        "open_credit": "Crédits actifs",
        "real_estate": "Prêts immobiliers",
        "debt_ratio": "Ratio d'endettement",
        "revolving": "Utilisation crédit renouvelable (%)",
        "late_30": "Retards 30-59 jours",
        "late_60": "Retards 60-89 jours",
        "late_90": "Retards ≥90 jours",
        "analysis_result": "Résultat de l'analyse",
        "default_prob": "Probabilité de défaut",
        "recommendation": "Recommandation",
        "low_risk": "Faible risque",
        "high_risk": "Risque élevé",
        "acceptance": "Acceptation",
        "rejection": "Rejet",
        "low_risk_badge": "✅ CLIENT FAIBLE RISQUE",
        "high_risk_badge": "⚠️ CLIENT À RISQUE ÉLEVÉ",
        "shap_section": "Analyse d'impact (SHAP)",
        "interpretation": "🗣️ Interprétation",
        "explanation_high": "**Risque élevé de défaut** principalement dû à :",
        "explanation_low": "**Client fiable avec faible risque** principalement dû à :",
        "conclusion_high": "🔍 **Conclusion** : Surveillance renforcée recommandée.",
        "conclusion_low": "✅ **Conclusion** : Profil sain, éligible au crédit.",
        "ai_report_section": "📋 Rapport d'Analyse Automatique",
        "generate_ai_report": "🤖 Générer rapport IA avancé",
        "missing_api_key": "⚠️ Clé API Cohere manquante dans st.secrets.",
        "ai_in_progress": "🔄 Analyse IA en cours...",
        "pdf_button": "📄 Télécharger PDF",
        "excel_button": "📊 Télécharger Excel",
        "pdf_title": "RiskScore Pro - Rapport IA",
        "page": "Page",
        "prompt_template": """
Analyse credit bancaire rapide pour ce client :

DONNÉES CLIENT :
- Âge: {age} ans
- Revenus: {income:,} FCFA/mois  
- Ratio endettement: {debt_ratio}
- Crédit renouvelable: {revolving}%
- Crédits actifs: {open_credit}
- Retards paiement: {late_30}/{late_60}/{late_90}
- Probabilité défaut: {proba:.1%}

CONSIGNE: Fournis une recommandation bancaire concise en 3 parties courtes:
1) PROFIL (2-3 mots sur le client)  
2) RISQUES (1-2 points principaux)
3) DÉCISION (Approuver/Refuser + 1 condition)

Max 80 mots, style professionnel bancaire.
"""
    },
    "en": {
        "page_title": "RiskScore Pro | Credit Analysis",
        "about": "Professional application for credit risk analysis based on AI.",
        "header_title": "RiskScore Pro - Credit Risk Assessment",
        "header_subtitle": "Predictive analytics platform powered by XGBoost, SHAP & Generative AI",
        "profile_client": "Client Profile",
        "age": "Age",
        "income": "Monthly income (FCFA)",
        "dependents": "Number of dependents",
        "open_credit": "Open credit lines",
        "real_estate": "Real estate loans",
        "debt_ratio": "Debt ratio",
        "revolving": "Revolving credit utilization (%)",
        "late_30": "30-59 days late payments",
        "late_60": "60-89 days late payments",
        "late_90": "≥90 days late payments",
        "analysis_result": "Analysis Result",
        "default_prob": "Probability of Default",
        "recommendation": "Recommendation",
        "low_risk": "Low risk",
        "high_risk": "High risk",
        "acceptance": "Approval",
        "rejection": "Rejection",
        "low_risk_badge": "✅ LOW-RISK CLIENT",
        "high_risk_badge": "⚠️ HIGH-RISK CLIENT",
        "shap_section": "Impact Analysis (SHAP)",
        "interpretation": "🗣️ Interpretation",
        "explanation_high": "**High risk of default** mainly due to:",
        "explanation_low": "**Reliable client with low risk** mainly due to:",
        "conclusion_high": "🔍 **Conclusion**: Enhanced monitoring recommended.",
        "conclusion_low": "✅ **Conclusion**: Healthy profile, eligible for credit.",
        "ai_report_section": "📋 Automatic Analysis Report",
        "generate_ai_report": "🤖 Generate advanced AI Report",
        "missing_api_key": "⚠️ Missing Cohere API key in st.secrets.",
        "ai_in_progress": "🔄 AI analysis in progress...",
        "pdf_button": "📄 Download PDF",
        "excel_button": "📊 Download Excel",
        "pdf_title": "RiskScore Pro - AI Report",
        "page": "Page",
        "prompt_template": """
Banking credit analysis for this client:

CLIENT DATA:
- Age: {age} years
- Income: {income:,} FCFA/month
- Debt ratio: {debt_ratio}
- Revolving credit: {revolving}%
- Active credits: {open_credit}
- Payment delays: {late_30}/{late_60}/{late_90}
- Default probability: {proba:.1%}

INSTRUCTION: Provide concise banking recommendation in 3 short parts:
1) PROFILE (2-3 words about client)
2) RISKS (1-2 main points)
3) DECISION (Approve/Reject + 1 condition)

Max 80 words, professional banking style.
"""
    }
}

# -------------------------------
# Détection langue via URL ?lang=fr ou ?lang=en
# -------------------------------
params = st.query_params
lang_candidate = params.get("lang", ["fr"])
if isinstance(lang_candidate, list):
    lang_candidate = lang_candidate[0].lower()
else:
    lang_candidate = lang_candidate.lower()
lang = lang_candidate if lang_candidate in ["fr", "en"] else "fr"
tr = T[lang]

# -------------------------------
# Configuration page
# -------------------------------
st.set_page_config(
    page_title=tr["page_title"],
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": tr["about"]}
)

# -------------------------------
# CSS thème sombre optimisé
# -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Variables CSS pour cohérence des couleurs sombres */
    :root {
        --primary-bg: #0a0e1a;
        --secondary-bg: #1a1f2e;
        --card-bg: #252c3d;
        --accent-bg: #2d3748;
        --input-bg: #1e2532;
        --primary-text: #ffffff;
        --secondary-text: #b0b9c6;
        --accent-color: #00d4ff;
        --success-color: #48bb78;
        --warning-color: #ed8936;
        --danger-color: #f56565;
        --border-color: #4a5568;
        --hover-bg: #364152;
    }
    
    * {
        font-family: 'Inter', sans-serif !important; 
        color: var(--primary-text) !important;
    }
    
    body, .stApp, .main, .block-container {
        background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%) !important;
        padding: 0.5rem !important; 
        margin: 0 !important; 
    }
    
    .block-container {
        max-width: 100% !important; 
        padding: 0.5rem !important; 
        margin: 0 !important;
    }
    
    /* Header moderne sombre */
    .header {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--accent-bg) 100%);
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 25px rgba(0, 212, 255, 0.15);
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .header h1 {
        color: var(--accent-color) !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    
    .header p {
        color: var(--secondary-text) !important;
        font-size: 1rem;
        margin-top: 0;
        font-weight: 500;
    }
    
    /* Conteneurs sombres améliorés */
    .metric-container {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--accent-bg) 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        text-align: center;
        box-shadow: 0 3px 20px rgba(0, 0, 0, 0.4);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 30px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.5);
    }
    
    /* Badges de risque avec thème sombre */
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        margin: 0.8rem auto;
        user-select: none;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .risk-low {
        background: linear-gradient(135deg, var(--success-color) 0%, #2f855a 100%);
        color: #f0fff4 !important;
        border-color: var(--success-color);
        box-shadow: 0 0 25px rgba(72, 187, 120, 0.5);
    }
    
    .risk-high {
        background: linear-gradient(135deg, var(--danger-color) 0%, #c53030 100%);
        color: #fff5f5 !important;
        border-color: var(--danger-color);
        box-shadow: 0 0 25px rgba(245, 101, 101, 0.5);
    }
    
    /* Titres de section sombres */
    .section-title {
        font-weight: 700;
        font-size: 1.3rem;
        margin: 1.2rem 0 1rem 0;
        color: var(--accent-color) !important;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 0 8px rgba(0, 212, 255, 0.3);
    }
    
    /* Graphiques SHAP avec fond sombre */
    .shap-plot {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    
    /* Cartes de contenu sombres */
    .card {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--accent-bg) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: var(--primary-text) !important;
        white-space: pre-wrap;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
        border: 1px solid var(--border-color);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* Formulaire compact sombre */
    .compact-form {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--accent-bg) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
        border: 1px solid var(--border-color);
    }
    
    /* Sliders sombres optimisés */
    .stSlider > div > div > div > div {
        background: var(--accent-color) !important;
        height: 4px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: var(--accent-color) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
    }
    
    .stSlider > div > div > div > div > div[role="slider"] {
        background-color: var(--accent-color) !important;
        border: 2px solid var(--primary-text) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    /* Inputs numériques sombres */
    .stNumberInput > div > div > input {
        background: var(--input-bg) !important;
        color: var(--primary-text) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 8px !important;
        height: 40px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Boutons sombres optimisés */
    button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-color) 0%, #0099cc 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        border: none !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00b8e6 0%, #0077a3 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Métriques Streamlit sombres */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--accent-bg) 100%) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: rgba(0, 212, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    div[data-testid="metric-container"] label {
        color: var(--secondary-text) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: var(--accent-color) !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 8px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Labels sombres */
    label {
        color: var(--primary-text) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Selectbox sombre */
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--primary-text) !important;
    }
    
    /* Scrollbar sombre */
    ::-webkit-scrollbar {
        width: 10px;
        background: var(--secondary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-color);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00b8e6;
    }
    
    /* Sidebar sombre */
    .css-1d391kg {
        background-color: var(--card-bg) !important;
    }
    
    /* Spinners et progress bars sombres */
    .stSpinner > div {
        border-top-color: var(--accent-color) !important;
    }
    
    /* Messages d'erreur sombres */
    .stAlert {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--primary-text) !important;
    }
    
    /* Améliorations diverses */
    .element-container {
        margin-bottom: 0.8rem !important;
    }
    
    /* Footer sombre */
    hr {
        border-color: var(--border-color) !important;
        margin: 2rem 0 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header moderne
# -------------------------------
st.markdown(f"""
<div class="header">
    <h1>{tr['header_title']}</h1>
    <p>{tr['header_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Layout principal en colonnes - TOUT EN TEMPS RÉEL
# -------------------------------
col_left, col_right = st.columns([1.2, 1], gap="medium")

with col_left:
    # Formulaire compact avec mise à jour temps réel
    with st.container():
        st.markdown(f'<div class="compact-form">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">👤 {tr["profile_client"]}</div>', unsafe_allow_html=True)
        
        # Sous-colonnes pour formulaire
        subcol1, subcol2 = st.columns(2, gap="small")
        
        with subcol1:
            age = st.slider(tr["age"], 18, 100, 35, key="age")
            income = st.number_input(tr["income"], min_value=0, value=1500000, step=50000, key="income")
            dependents = st.slider(tr["dependents"], 0, 10, 2, key="dependents")
            open_credit = st.slider(tr["open_credit"], 0, 30, 5, key="open_credit")
            real_estate = st.slider(tr["real_estate"], 0, 10, 1, key="real_estate")
        
        with subcol2:
            debt_ratio = st.slider(tr["debt_ratio"], 0.0, 10.0, 0.5, step=0.1, key="debt_ratio")
            revolving = st.slider(tr["revolving"], 0.0, 100.0, 30.0, step=1.0, key="revolving")
            late_30 = st.slider(tr["late_30"], 0, 10, 0, key="late_30")
            late_60 = st.slider(tr["late_60"], 0, 10, 0, key="late_60")
            late_90 = st.slider(tr["late_90"], 0, 10, 0, key="late_90")
        
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Calculs optimisés avec cache - MIS À JOUR EN TEMPS RÉEL
# -------------------------------
client_df, proba, classe = predict_client(age, income, dependents, open_credit, real_estate, debt_ratio, revolving, late_30, late_60, late_90)
shap_values, shap_df = compute_shap_values(age, income, dependents, open_credit, real_estate, debt_ratio, revolving, late_30, late_60, late_90)

with col_right:
    # Résultats en temps réel
    st.markdown(f'<div class="section-title">📊 {tr["analysis_result"]}</div>', unsafe_allow_html=True)
    
    # Métriques principales
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(tr["default_prob"], f"{proba:.1%}", delta=None)
    with metric_col2:
        status_text = tr["acceptance"] if classe == 0 else tr["rejection"]
        delta_text = tr["low_risk"] if classe == 0 else tr["high_risk"]
        st.metric(tr["recommendation"], status_text, delta_text)

    # Badge de risque dynamique
    if classe == 0:
        st.markdown(f'<div style="text-align:center;"><div class="risk-badge risk-low">{tr["low_risk_badge"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align:center;"><div class="risk-badge risk-high">{tr["high_risk_badge"]}</div></div>', unsafe_allow_html=True)

    # Revenu formaté - MIS À JOUR EN TEMPS RÉEL
    st.markdown(f'<div class="metric-container"><div class="metric-label">{tr["income"]}</div><div class="metric-value">{income:,} FCFA</div></div>', unsafe_allow_html=True)

# -------------------------------
# Section SHAP optimisée - TEMPS RÉEL
# -------------------------------
st.markdown(f'<div class="section-title">📈 {tr["shap_section"]}</div>', unsafe_allow_html=True)

shap_col1, shap_col2 = st.columns([1.3, 1], gap="medium")

with shap_col1:
    with st.container():
        st.markdown('<div class="shap-plot">', unsafe_allow_html=True)
        
        # Graphique SHAP optimisé - MIS À JOUR EN TEMPS RÉEL
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#252c3d')
        ax.set_facecolor('#252c3d')
        
        # Utiliser directement les valeurs SHAP cachées
        shap.plots.waterfall(shap_values, max_display=6, show=False)
        plt.title("Impact des variables sur la prédiction", color='white', fontsize=11, fontweight='bold')
        
        # Personnaliser les couleurs pour le thème sombre
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()  # LIBÉRATION MÉMOIRE
        st.markdown('</div>', unsafe_allow_html=True)

with shap_col2:
    # Interprétation optimisée avec données cachées - TEMPS RÉEL
    st.markdown(f"### {tr['interpretation']}")
    
    top_features = shap_df.head(3)
    
    if classe == 1:
        st.markdown(tr["explanation_high"])
    else:
        st.markdown(tr["explanation_low"])
    
    for _, row in top_features.iterrows():
        impact_direction = "↗️ augmente" if row['shap_value'] > 0 else "↘️ diminue"
        feature_display = row['feature'].replace('RevolvingUtilizationOfUnsecuredLines', 'Utilisation crédit renouvelable')
        feature_display = feature_display.replace('NumberOfTime30-59DaysPastDueNotWorse', 'Retards 30-59 jours')
        feature_display = feature_display.replace('NumberOfTime60-89DaysPastDueNotWorse', 'Retards 60-89 jours')
        feature_display = feature_display.replace('NumberOfTimes90DaysLate', 'Retards ≥90 jours')
        feature_display = feature_display.replace('MonthlyIncome', 'Revenu mensuel')
        feature_display = feature_display.replace('NumberOfOpenCreditLinesAndLoans', 'Crédits actifs')
        feature_display = feature_display.replace('NumberRealEstateLoansOrLines', 'Prêts immobiliers')
        feature_display = feature_display.replace('NumberOfDependents', 'Personnes à charge')
        feature_display = feature_display.replace('DebtRatio', 'Ratio d\'endettement')
        feature_display = feature_display.replace('age', 'Âge')
        
        st.markdown(f"• **{feature_display}** : {impact_direction} le risque")
    
    if classe == 1:
        st.markdown(tr["conclusion_high"])
    else:
        st.markdown(tr["conclusion_low"])

# -------------------------------
# Section IA optimisée - RAPPORT AUTOMATIQUE EN TEMPS RÉEL
# -------------------------------
st.markdown(f'<div class="section-title">📋 {tr["ai_report_section"]}</div>', unsafe_allow_html=True)

ai_col1, ai_col2 = st.columns([2, 1], gap="medium")

with ai_col1:
    # Rapport automatique en temps réel basé sur les données actuelles
    auto_report = generate_auto_ai_report(age, income, dependents, open_credit, real_estate, 
                                         debt_ratio, revolving, late_30, late_60, late_90, 
                                         proba, classe, shap_df, lang)
    
    st.markdown(f'<div class="card">{auto_report}</div>', unsafe_allow_html=True)
    
    # Option pour rapport IA avancé avec Cohere
    st.markdown("---")
    
    # PROMPT avec valeurs actuelles - optimisé pour être plus court
    prompt = tr["prompt_template"].format(
        age=age,
        income=income,
        debt_ratio=debt_ratio,
        revolving=revolving,
        open_credit=open_credit,
        real_estate=real_estate,
        dependents=dependents,
        late_30=late_30,
        late_60=late_60,
        late_90=late_90,
        proba=proba
    )

    if "texte_ia" not in st.session_state:
        st.session_state["texte_ia"] = None

    if st.button(tr["generate_ai_report"], key="ai_button"):
        if "COHERE_API_KEY" not in st.secrets:
            st.error(tr["missing_api_key"])
        else:
            with st.spinner(tr["ai_in_progress"]):
                try:
                    co = cohere.Client(st.secrets["COHERE_API_KEY"])
                    response = co.generate(
                        model="command-r-plus",
                        prompt=prompt,
                        max_tokens=120,  # Réduit pour des recommandations plus courtes
                        temperature=0.2   # Réduit pour plus de consistance
                    )
                    st.session_state["texte_ia"] = response.generations[0].text.strip()
                except Exception as e:
                    st.error(f"Erreur API Cohere: {str(e)}")

    if st.session_state["texte_ia"]:
        st.markdown("### 🤖 Rapport IA Avancé")
        st.markdown(f'<div class="card">{st.session_state["texte_ia"]}</div>', unsafe_allow_html=True)

with ai_col2:
    # Boutons de téléchargement optimisés
    st.markdown("### 📥 Téléchargements")
    
    # Utiliser le rapport automatique pour les téléchargements
    report_to_export = st.session_state["texte_ia"] if st.session_state["texte_ia"] else auto_report
    
    # Génération PDF optimisée
    try:
        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 10, remove_accents(tr["pdf_title"]), ln=True, align="C")
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"{tr['page']} {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "", 10)
        
        # Ajouter les données du client
        pdf.cell(0, 8, f"PROFIL CLIENT - {age} ans", ln=True, align="L")
        pdf.cell(0, 6, f"Revenu: {income:,} FCFA", ln=True)
        pdf.cell(0, 6, f"Probabilite de defaut: {proba:.1%}", ln=True)
        pdf.cell(0, 6, f"Recommandation: {tr['acceptance'] if classe == 0 else tr['rejection']}", ln=True)
        pdf.ln(5)
        
        # Ajouter le rapport
        for line in report_to_export.split('\n'):
            clean_line = remove_accents(line.strip())
            if clean_line:
                pdf.multi_cell(180, 6, clean_line)
                pdf.ln(2)
        
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output = io.BytesIO(pdf_bytes)
        pdf_output.seek(0)
        
        st.download_button(
            label=tr["pdf_button"],
            data=pdf_output,
            file_name=f"rapport_credit_{age}ans_{proba:.0%}_risque.pdf",
            mime="application/pdf",
            key="pdf_download"
        )
    except Exception as e:
        st.error(f"Erreur génération PDF: {str(e)}")

    # Génération Excel optimisée
    try:
        df_export = pd.DataFrame([{
            "Date_Analyse": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            tr["age"]: age,
            tr["income"]: income,
            tr["debt_ratio"]: debt_ratio,
            tr["revolving"]: revolving,
            tr["open_credit"]: open_credit,
            tr["real_estate"]: real_estate,
            tr["dependents"]: dependents,
            tr["late_30"]: late_30,
            tr["late_60"]: late_60,
            tr["late_90"]: late_90,
            tr["default_prob"]: f"{proba:.1%}",
            tr["recommendation"]: tr["acceptance"] if classe == 0 else tr["rejection"],
            "Niveau_Risque": tr["low_risk"] if classe == 0 else tr["high_risk"]
        }])
        
        # Ajouter les facteurs SHAP
        shap_export = shap_df.head(5).copy()
        shap_export['feature'] = shap_export['feature'].replace({
            'RevolvingUtilizationOfUnsecuredLines': 'Utilisation_Credit_Renouvelable',
            'NumberOfTime30-59DaysPastDueNotWorse': 'Retards_30_59_jours',
            'NumberOfTime60-89DaysPastDueNotWorse': 'Retards_60_89_jours',
            'NumberOfTimes90DaysLate': 'Retards_90_jours_plus',
            'MonthlyIncome': 'Revenu_Mensuel',
            'NumberOfOpenCreditLinesAndLoans': 'Credits_Actifs',
            'NumberRealEstateLoansOrLines': 'Prets_Immobiliers',
            'NumberOfDependents': 'Personnes_Charge',
            'DebtRatio': 'Ratio_Endettement',
            'age': 'Age'
        })
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name="Resultat_Client")
            shap_export.to_excel(writer, index=False, sheet_name="Facteurs_Impact")
        excel_buffer.seek(0)

        st.download_button(
            label=tr["excel_button"],
            data=excel_buffer,
            file_name=f"analyse_credit_{age}ans_{proba:.0%}_risque.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="excel_download"
        )
    except Exception as e:
        st.error(f"Erreur génération Excel: {str(e)}")

    # Score de santé financière simplifié
    st.markdown("### 📊 Score de Santé")
    
    score_sante = (1 - proba) * 100
    if score_sante >= 80:
        score_color = "🟢"
        score_text = "Excellent"
    elif score_sante >= 60:
        score_color = "🟡"
        score_text = "Correct"
    else:
        score_color = "🔴"
        score_text = "Faible"
    
    st.markdown(f'<div class="metric-container"><div class="metric-label">Score Financier</div><div class="metric-value">{score_color} {score_sante:.0f}/100</div><div style="color: var(--secondary-text); font-size: 0.8rem; margin-top: 0.3rem;">{score_text}</div></div>', unsafe_allow_html=True)

# -------------------------------
# Footer simplifié
# -------------------------------
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown(f"**⚡ Analyse en temps réel** - Mise à jour automatique")

with footer_col2:
    st.markdown(f"**🎯 Modèle XGBoost** - Précision optimisée")

# -------------------------------
# Debug optionnel dans la sidebar
# -------------------------------
if st.sidebar.checkbox("🔧 Infos Debug", value=False):
    st.sidebar.markdown("### 🚀 Performance")
    st.sidebar.markdown(f"**Modèle:** ✅ Chargé")
    st.sidebar.markdown(f"**Cache SHAP:** ✅ Actif")
    st.sidebar.markdown(f"**Proba:** {proba:.3f}")
    st.sidebar.markdown(f"**Classe:** {classe}")
    st.sidebar.markdown(f"**Top facteur:** {shap_df.iloc[0]['feature'][:20]}...")
    st.sidebar.markdown(f"**Impact:** {shap_df.iloc[0]['shap_value']:.3f}")
    
    # Données actuelles
    st.sidebar.markdown("### 📊 Données")
    st.sidebar.json({
        "age": age,
        "income_fcfa": f"{income:,}",
        "probability": f"{proba:.1%}",
        "risk": "HIGH" if classe == 1 else "LOW"
    })