#%%
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from rdkit import Chem
from smiles_processing import clean_smiles_batch, segment_smiles, segment_smiles_corpus, apply_label_encoding
from SmilesEnumerator import SmilesEnumerator
from DeepCocrystal import DeepCrystal

API_LEN = 80
COF_LEN = 80
N_AUGMENT = 10

with open("token2label.json", "r") as f:
    token2label = json.load(f)

predictor_params = {
    "n_cnn": 2,
    "n_filters": 256,
    "kernel_sizes": 4,
    "cnn_activation": "selu",
    "n_dense": 3,
    "dense_layer_size": 1024,
    "dense_activation": "relu",
    "embedding_dim": 32,
    "dropout_rate": 0.1,
    "optimizer_name": "adam",
    "learning_rate": 0.05,
    "batch_size": 512,
    "n_epochs": 500,
    "csv_log_path": "./results.csv"
}

loaded_model = tf.keras.models.load_model("DeepCrystal_trained")
deep_cocrys = DeepCrystal(**predictor_params)
deep_cocrys.model = loaded_model

# SMILES cleaning and randomization
sme = SmilesEnumerator()

def clean_and_filter(smiles_list, length_threshold=80):
    cleaned = clean_smiles_batch(
        smiles_list,
        remove_salt=False,
        desalt=False,
        uncharge=True,
        sanitize=False,
        remove_stereochemistry=True,
        to_canonical=True,
    )
    return [c if c is not None and len(segment_smiles(c)) <= length_threshold else None for c in cleaned]

def randomize_smiles(smiles, n=10):
    return [sme.randomize_smiles(smiles, iteration=i) for i in range(n)]

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# Title and description for the app
st.set_page_config(page_title="DeepCocrystal", page_icon="💊", layout="wide")
st.title("DeepCocrystal")
st.write("""
Welcome to DeepCocrystal, a predictive model that will help you select promising coformers for
your co-crystallization screenings.
         
Through this online interface, you can predict the probability of two molecules co-crystallizing, along with their predictive uncertainty estimated through SMILES randomization — useful for prioritizing your lab experiments.

You can read more details in our [publication](https://chemrxiv.org/engage/chemrxiv/article-details/66704f0501103d79c56770b2).

**✒️ Enter API-Coformer SMILES pairs manually**
""")


# Entry list (Session state to store across reruns)
if 'entries' not in st.session_state:
    st.session_state.entries = []

if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# Manual entry form
with st.form("entry_form"):
    api_input = st.text_input("API SMILES:")
    cof_input = st.text_input("Coformer SMILES:")
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        if not api_input or not cof_input:
            st.warning("Both SMILES must be provided.")
        elif not is_valid_smiles(api_input) or not is_valid_smiles(cof_input):
            st.warning("Invalid SMILES entered. Please check the format.")
        else:
            st.session_state.entries.append((api_input, cof_input))
            st.success("Pair added.")

# CSV Upload
st.write('**⬆️ *and/or* Upload a CSV file with the SMILES of the molecular pairs to be predicted**')

load_sample = st.checkbox("If you want to upload an example of data loading, check this box!")
if load_sample:
    try:
        sample_df = pd.read_csv('./data/example.csv')
        if "API" in sample_df.columns and "Coformer" in sample_df.columns:
            st.session_state.entries = list(sample_df[["API", "Coformer"]].itertuples(index=False, name=None))
            st.success("Example loaded")
        else:
            st.error("The sample file doesn't contain 'API' and 'Coformer' columns.")
    except FileNotFoundError:
        st.error("Sample file 'example.csv' not found.")


st.markdown("""
    **Example CSV format:**<br>
    The file should have two columns: `API` and `Coformer`, with SMILES strings (up to 80 tokens) as values.<br>
    ```csv
    API,Coformer
    CCCc1cc(C(N)=S)ccn1,O=S(=O)(O)CCS(=O)(=O)O
    O=C(O)c1cc(O)ccc1O,Nc1ncnc2nc[nH]c12
    ```
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file", type="csv")
csv_df = None

if uploaded_file:
    try:
        csv_df = pd.read_csv(uploaded_file)
        if "API" not in csv_df.columns or "Coformer" not in csv_df.columns:
            st.error("CSV must contain 'API' and 'Coformer' columns.")
            csv_df = None
        else:
            st.success(f"{len(csv_df)} entries loaded from CSV.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Combine entries from manual and CSV
all_entries = st.session_state.entries.copy()
if csv_df is not None:
    all_entries.extend(list(csv_df[["API", "Coformer"]].itertuples(index=False, name=None)))

# Show current list
if all_entries:
    st.write("**📍 Data Loaded**")
    df_pairs = pd.DataFrame(all_entries, columns=["API", "Coformer"])
    st.dataframe(df_pairs)

    if st.button("🔮 Run predictions for all the entries"):
        results = []
        for api, cof in all_entries:
            api_cleaned = clean_and_filter([api])[0]
            cof_cleaned = clean_and_filter([cof])[0]

            if api_cleaned is None or cof_cleaned is None:
                results.append({
                    "API": api,
                    "Coformer": cof,
                    "Prediction": "Invalid SMILES",
                    "Std Dev": "-"
                })
                continue

            rand_api = randomize_smiles(api_cleaned, N_AUGMENT)
            rand_cof = randomize_smiles(cof_cleaned, N_AUGMENT)

            seg_api = segment_smiles_corpus(rand_api)
            seg_cof = segment_smiles_corpus(rand_cof)

            encoded_api = apply_label_encoding(seg_api, token2label)
            encoded_cof = apply_label_encoding(seg_cof, token2label)

            padded_api = tf.keras.preprocessing.sequence.pad_sequences(
                encoded_api, maxlen=API_LEN, padding='post', value=0
            )
            padded_cof = tf.keras.preprocessing.sequence.pad_sequences(
                encoded_cof, maxlen=COF_LEN, padding='post', value=0
            )

            predictions = deep_cocrys.predict(padded_api, padded_cof).flatten()
            avg_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            results.append({
                "API": api,
                "Coformer": cof,
                "Prediction": f"{avg_pred:.4f}",
                "Std Dev": f"{std_pred:.4f}"
            })

        st.session_state.result_df = pd.DataFrame(results)
        st.success("Predictions completed!")

    if st.session_state.result_df is not None:
        st.write("**✔️ DeepCocrystal Predictions**")
        st.dataframe(st.session_state.result_df)
        st.download_button(
            "📄 Download results as CSV",
            st.session_state.result_df.to_csv(index=False),
            file_name="DeepCocrystal_predictions.csv"
        )
else:
    st.info("No SMILES pairs yet. Use the form or upload a CSV.")



st.write("📖 **Cite DeepCocrystal**")
st.write("""
If DeepCocrystal has been useful for driving your experimental tests, please cite us:
""")

st.code("""
@article{birolo2024deep,
  title={Deep Supramolecular Language Processing for Co-crystal Prediction},
  author={Birolo, Rebecca and {\"O}z{\c{c}}elik, R{\i}za and Aramini, Andrea and Gobetto, Roberto and Chierotti, Michele Remo and Grisoni, Francesca},
  year={2024}
}
""", language='bibtex')
# %%
