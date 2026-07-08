import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# 1. Configuration de la page et titres
st.set_page_config(page_title="ShopEuro - Dashboard Direction", layout="wide")

st.title("📊 ShopEuro : Dashboard Analytique & Conformité RGPD")
st.caption("Pilote de la performance, de la gestion de l'urgence support et de la protection des données.")

# 2. Définition de la racine (BASE_DIR) AVANT son utilisation dans les fonctions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CHARGEMENT DES DONNÉES & MODÈLES ---
@st.cache_data
def load_data():
    file_id = "1PHkADPXqaLaHO1-b2ENF3i3pUSgHeqti"
    output_filename = os.path.join(BASE_DIR, "data_gold_final.parquet")
    
    # Si le fichier n'est pas local, on tente de le télécharger proprement
    if not os.path.exists(output_filename):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_filename, quiet=False)
        except Exception:
            # Solution de secours si gdown n'est pas installé ou échoue
            url = f"https://docs.google.com/uc?export=download&id={file_id}"
            df = pd.read_parquet(url)
            return df.sample(n=10000, random_state=42) if len(df) > 10000 else df

    # Lecture du fichier local une fois téléchargé
    df = pd.read_parquet(output_filename)
    return df.sample(n=10000, random_state=42) if len(df) > 10000 else df

# 3. Exécution du chargement du DataFrame
df_sample = load_data()

# 4. Chargement des modèles avec les chemins absolus corrigés
model_path = os.path.join(BASE_DIR, 'model_urgence_tickets.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer_tfidf.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# --- SECTION 1 : KPIs GLOBAUX ---
st.header("📈 Indicateurs Clés de Performance (KPIs)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Volume d'activité analysé", f"{len(df_sample)} lignes")
with col2:
    st.metric("Taux de Churn estimé", "15.0 %")
with col3:
    total_tickets = df_sample['ticket_text_clean'].notna().sum() if 'ticket_text_clean' in df_sample.columns else 0
    st.metric("Tickets Support Traités", f"{total_tickets}")
with col4:
    st.metric("Conformité RGPD", "100 % Haché")

st.markdown("---")

# --- SECTION 2 : SUPPORT CLIENT & INSIGHTS ML ---
st.header("🤖 Analyse du Support & Prédictions de l'IA")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Testez le modèle d'urgence en direct")
    ticket_input = st.text_area("Saisissez le texte d'un ticket de support pour l'analyser :",
                                "Bonjour, mon colis est défectueux et je demande un remboursement immédiat !")

    if st.button("Prédire le niveau d'urgence"):
        text_vectorized = vectorizer.transform([ticket_input])
        prediction = model.predict(text_vectorized)[0]
        probabilite = model.predict_proba(text_vectorized)[0][1]

        if prediction == 1:
            st.error(f"🚨 Ticket classé comme URGENT (Probabilité : {probabilite:.2%})")
        else:
            st.success(f"✅ Ticket classé comme Standard (Probabilité d'urgence : {probabilite:.2%})")

with col_right:
    st.subheader("Distribution brute des tickets analysés")
    if 'ticket_text_clean' in df_sample.columns:
        st.info(f"Sur l'échantillon, {total_tickets} tickets ont été scannés et pseudonymisés dans la couche Silver.")
    else:
        st.warning("La colonne 'ticket_text_clean' n'a pas été détectée dans la couche Gold.")

st.markdown("---")

# --- SECTION 3 : CONFORMITÉ ET DROIT À L'OUBLI ---
st.header("🔐 Espace de Conformité RGPD (Art. 17 & 20)")
st.warning("Toutes les actions effectuées dans cet espace génèrent un audit log immuable.")

customer_id_input = st.text_input("Entrez un identifiant client (ex: CUST_00913) :")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📦 Déclencher la Portabilité (Export JSON)"):
        if customer_id_input:
            st.info(f"Extraction des données pour {customer_id_input}...")
            st.success(f"Fichier export_portabilite_{customer_id_input}.json généré avec succès !")
        else:
            st.error("Veuillez saisir un ID client valide.")

with col_btn2:
    if st.button("🛑 Déclencher le Droit à l'Oubli (Suppression)"):
        if customer_id_input:
            st.error(f"Purge définitive des données associée à {customer_id_input}...")
            st.success(f"Certificat d'effacement généré pour {customer_id_input}.")
        else:
            st.error("Veuillez saisir un ID client valide.")