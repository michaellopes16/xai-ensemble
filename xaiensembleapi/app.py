# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd

# --- Funções da explicabilidade ---

def perturb_block(sequence, start, end, mode='zero'):
    perturbed = sequence.copy()
    if mode == 'zero':
        perturbed[start:end] = 0
    elif mode == 'smooth':
        perturbed[start:end] = np.mean(sequence[max(0, start-5):min(len(sequence), end+5)])
    return perturbed

def explain_inceptiontime(model, sequence, num_blocks=20, perturb_mode='zero'):
    baseline_pred = model.predict(sequence)
    baseline_class = np.argmax(baseline_pred)
    
    T = len(sequence)
    block_size = T // num_blocks
    importances = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < num_blocks - 1 else T
        
        perturbed_seq = perturb_block(sequence, start, end, mode=perturb_mode)
        perturbed_pred = model.predict(perturbed_seq)
        
        delta = abs(baseline_pred[baseline_class] - perturbed_pred[baseline_class])
        importances[i] = delta
    
    importances /= importances.max()
    return importances

def plot_explanation(sequence, importances, num_blocks=20):
    T = len(sequence)
    block_size = T // num_blocks
    colors = np.repeat(importances, block_size)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(range(T), sequence, c=colors[:T], cmap='hot', s=20)
    fig.colorbar(scatter, ax=ax, label='Importance')
    ax.set_title('Explicabilidade baseada em perturbação')
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Valor')
    st.pyplot(fig)

# --- Modelo Simulado ---

class FakeInceptionTimeModel:
    def predict(self, X):
        features = np.array([X.mean(), X.max(), X.std()])
        logits = features @ np.array([0.3, 0.5, -0.2])
        prob = 1 / (1 + np.exp(-logits))
        return np.array([1 - prob, prob])

# --- Interface Streamlit ---

st.title('🕒 Explicabilidade de Séries Temporais com Perturbação')

uploaded_file = st.file_uploader("Faça upload da série temporal (.csv ou .npy):", type=["csv", "npy"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        sequence = df.values.squeeze()
    else:
        sequence = np.load(uploaded_file)
    
    st.write("✅ Série carregada! Tamanho:", len(sequence))
    
    num_blocks = st.slider('Número de blocos:', min_value=5, max_value=50, value=20)
    perturb_mode = st.selectbox('Modo de perturbação:', ['zero', 'smooth'])
    
    if st.button('Executar explicabilidade'):
        model = FakeInceptionTimeModel()
        importances = explain_inceptiontime(model, sequence, num_blocks=num_blocks, perturb_mode=perturb_mode)
        
        st.write("✅ Explicabilidade executada!")
        plot_explanation(sequence, importances, num_blocks=num_blocks)
        
        # Download CSV
        importance_df = pd.DataFrame({
            'Bloco': np.arange(1, num_blocks + 1),
            'Importância': importances
        })
        
        csv = importance_df.to_csv(index=False)
        st.download_button("📥 Baixar Importâncias (.csv)", data=csv, file_name='importancias.csv', mime='text/csv')

else:
    st.info('Por favor, faça upload de um arquivo de série temporal.')
