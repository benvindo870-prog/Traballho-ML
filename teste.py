import streamlit as st
import pandas as pd 
import joblib 
import numpy as np

st.set_page_config(page_title="Previsão de preço de casas", layout="wide")

if "historico" not in st.session_state:
    st.session_state.historico = []
if "regioes_eliminadas" not in st.session_state:
    st.session_state.regioes_eliminadas = []

COORDENADAS_PT = {
    'Lisboa': (38.7223, -9.1393),
    'Porto': (41.1579, -8.6291),
    'Braga': (41.5503, -8.4201),
    'Coimbra': (40.2033, -8.4103),
    'Faro': (37.0179, -7.9308),
    'Setúbal': (38.5244, -8.8931),
    'Zona Geral': (39.3999, -8.2245)
}

st.title("🏠 Simulador Avançado de Preços de Imóveis")
st.write("Introduza os parâmetros para calcular previsões baseadas em modelos regularizados (Ridge/Lasso).")

tipo_negocio = st.radio("O que deseja simular?", ["Arrendamento (Renda)", "Compra / Venda (Preço Total)"], horizontal=True)

if tipo_negocio == "Arrendamento (Renda)":
    ficheiro_modelo = 'modelo_arrendamento.pkl'
    sinal_moeda = "€ / mês"
else:
    ficheiro_modelo = 'modelo_venda.pkl'
    sinal_moeda = "€"

@st.cache_resource 
def carregar_modelo_dinamico(ficheiro):
    return joblib.load(ficheiro)

try:
    dados = carregar_modelo_dinamico(ficheiro_modelo)
    modelo = dados['modelo']
    colunas_treino = dados['colunas_treino']
    localizacoes_base = dados.get('localizacoes', ["Zona Geral"])
    df_historico = dados.get('df_historico', pd.DataFrame())
    
    localizacoes = [l for l in localizacoes_base if l not in st.session_state.regioes_eliminadas]
    if not localizacoes: localizacoes = ["Zona Geral"]

    col_esquerda, col_direita = st.columns([1, 1])

    with col_esquerda:
        st.subheader("📋 Características do Imóvel")
        with st.form("formulario_previsao"):
            area = st.number_input("Área Útil (m²)", min_value=10, max_value=1000, value=75, step=5, format="%d")
            quartos = st.number_input("Número de Quartos (0 para T0/Estúdio)", min_value=0, max_value=10, value=2, step=1, format="%d")
            
            st.write("---")
            st.markdown("**Localização do Imóvel**")
            
            localizacao_digitada = st.text_input("✍️ Digite ou pesquise a localização (Opcional):", placeholder="Ex: Lisboa")
            
            default_index = 0
            if localizacao_digitada:
                matches = [i for i, x in enumerate(localizacoes) if localizacao_digitada.lower() in x.lower()]
                if matches: default_index = matches[0]

            localizacao_selecionada = st.selectbox("📍 Selecione a Localização confirmada:", options=localizacoes, index=default_index)
            botao_prever = st.form_submit_button("🔮 Calcular Preço Estimado")

        st.subheader("🗑️ Gestão de Regiões")
        regiao_para_remover = st.selectbox("Escolha uma região para ocultar do simulador:", options=["---"] + localizacoes)
        if st.button("Eliminar Região selecionada") and regiao_para_remover != "---":
            if regiao_para_remover not in st.session_state.regioes_eliminadas:
                st.session_state.regioes_eliminadas.append(regiao_para_remover)
                st.rerun()
        
        if st.session_state.regioes_eliminadas:
            if st.button("Restaurar todas as regiões eliminadas"):
                st.session_state.regioes_eliminadas = []
                st.rerun()

    with col_direita:
        if botao_prever:
            dados_input = {
                'area_m2': int(area),
                'quartos': int(quartos),
                'area_por_quarto': int(area / (quartos + 1)) 
            }
            

            for col in colunas_treino:
                if col.startswith('localizacao_'):
                    dados_input[col] = 0
            
            if f'localizacao_{localizacao_selecionada}' in colunas_treino:
                dados_input[f'localizacao_{localizacao_selecionada}'] = 1
            
            df_predict = pd.DataFrame([dados_input])
            df_predict_processado = df_predict.reindex(columns=colunas_treino, fill_value=0)
            
            preco_estimado = modelo.predict(df_predict_processado)[0]
            if preco_estimado < 0: preco_estimado = 0.0
            
            if preco_estimado.is_integer():
                st.success("Preço Estimado" f"{int(preco_estimado):,} {sinal_moeda}".replace(",", " ") if preco_estimado.is_integer() else f"{preco_estimado:,.2f} {sinal_moeda}")
            else:
                st.success(f"### Preço Estimado: **{preco_estimado:,.2f} {sinal_moeda}**")
            
            st.session_state.historico.append({
                "Negócio": tipo_negocio,
                "Localização": localizacao_selecionada,
                "Área (m²)": area,
                "Quartos": quartos,
                "Preço Estimado": f"{preco_estimado:.2f} {sinal_moeda}"
            })
            st.toast("Simulação guardada com sucesso!")

        st.subheader("🗺️ Localização Geográfica do Investimento")
        lat, lon = COORDENADAS_PT.get(localizacao_selecionada, COORDENADAS_PT['Zona Geral'])
        df_mapa = pd.DataFrame({'lat': [lat + np.random.uniform(-0.01, 0.01)], 'lon': [lon + np.random.uniform(-0.01, 0.01)]})
        st.map(df_mapa, zoom=11 if localizacao_selecionada != 'Zona Geral' else 6)


    st.markdown("---")
    aba1, aba2 = st.tabs(["📊 Comparador de Preços Inter-regiões", "💾 Histórico e Exportação"])
    
    with aba1:
        st.subheader("⚔️ Comparação de Preço Médio Anunciado")
        if not df_historico.empty:
            df_agrupado = df_historico.groupby('localizacao')['preco'].mean().reset_index()
            df_agrupado.columns = ['Localização', 'Preço Médio Registado (€)']
            
            st.dataframe(
                df_agrupado.style.highlight_max(subset=['Preço Médio Registado (€)'], color='#ff4b4b')
                .highlight_min(subset=['Preço Médio Registado (€)'], color='#24a148'), 
                use_container_width=True
            )
        else:
            st.info("Sem dados estatísticos guardados no modelo para gerar comparações.")

    with aba2:
        st.subheader("📋 Simulações Guardadas nesta Sessão")
        if st.session_state.historico:
            df_hist_visualizar = pd.DataFrame(st.session_state.historico)
            st.dataframe(df_hist_visualizar, use_container_width=True)
            
            csv = df_hist_visualizar.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descarregar Histórico de Simulações (CSV)",
                data=csv,
                file_name="simulacoes_imobiliarias.csv",
                mime="text/csv"
            )
        else:
            st.caption("Ainda não efetuou nenhuma previsão para guardar nesta sessão.")

except FileNotFoundError:
    st.error(f"Erro: O ficheiro '{ficheiro_modelo}' não foi encontrado. Por favor, execute o script de treino primeiro para gerar os novos modelos.")
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar os dados na interface: {e}")