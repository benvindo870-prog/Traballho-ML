# Sistema de Previsão de Preços de Imóveis com Machine Learning

**Unidade Curricular:** Inteligência Artificial Aplicada  
**Projeto:** Sistema de previsão de imóveis com Machine Learning  

**Alunos** Benvindo Elias nº a22510991  
Sadjo Djalo nº a22502320  
João Francisco nº a22510064  

**Docente** Pedro Sobreiro  

**Ano Letivo:** 2025/2026 | **Entrega:** Junho de 2026  

---

## Resumo

Sistema avançado de previsão de preços de imóveis baseado em Machine Learning, com dados recolhidos por web scraping do portal OLX Portugal. O ecossistema foi evoluído para suportar dois pipelines independentes (Arrendamento e Compra/Venda) com seleção dinâmica de modelos regularizados (Regressão Linear, Ridge e Lasso). A solução inclui uma interface web enriquecida em Streamlit que oferece estimativas em tempo real, geolocalização por mapa dinâmico, gestão ativa de regiões e exportação do histórico de simulações. As principais tecnologias são Python, pandas, scikit-learn, joblib e Streamlit.

---

## 1. Introdução

### 1.1 Contextualização e Problema

O mercado imobiliário português tem registado volatilidade crescente, dificultando a avaliação justa de propriedades. Sem ferramentas acessíveis, compradores e vendedores dependem de avaliadores certificados ou comparações manuais — processos morosos e subjetivos. 

**Questão central:** Dado um conjunto de características mensuráveis de um imóvel (incluindo a sua localização), é possível estimar automaticamente o seu valor justo com base em dados históricos através de modelos preditivos otimizados?

### 1.2 Objetivos

| Tipo | Objetivo |
|---|---|
| **Geral** | Desenvolver um simulador avançado e dinâmico de previsão imobiliária acessível via interface web. |
| **Específico 1** | Implementar uma extração inteligente e tratamento segregado para dados de Arrendamento e Venda. |
| **Específico 2** | Incorporar a localização geográfica como variável categórica do modelo através de *One-Hot Encoding*. |
| **Específico 3** | Automatizar a seleção do melhor modelo preditivo avaliando abordagens lineares e regularizadas (*Ridge* e *Lasso*). |
| **Específico 4** | Disponibilizar ferramentas de usabilidade na UI (filtros regionais, mapas interativos e exportação de dados). |

---

## 2. Estado da Arte

### 2.1 Conceitos Fundamentais

* **Aprendizagem Supervisionada:** O modelo aprende a partir de dados históricos rotulados (características $\rightarrow$ preço) para prever novos cenários.
* **Regressão e Regularização:** Além da tradicional Regressão Linear, utilizam-se técnicas de penalização (*Ridge/L2* e *Lasso/L1*) para mitigar o problema de multicolinearidade e evitar o *overfitting* provocado pela introdução de múltiplas variáveis de localização.
* **Feature Engineering:** Criação de métricas derivadas como a `area_por_quarto`, que auxiliam o modelo a capturar a proporção do espaço físico face à tipologia do imóvel.
* **One-Hot Encoding:** Abordagem estatística utilizada para converter variáveis de texto (Localizações) em colunas binárias ($0$ ou $1$), permitindo a sua leitura por algoritmos matemáticos.

### 2.2 Tecnologias Utilizadas

| Tecnologia | Função |
|---|---|
| **Python 3.9+** | Linguagem e motor lógico do ecossistema. |
| **pandas** | Limpeza avançada, tratamento por expressões regulares (regex) e engenharia de atributos. |
| **scikit-learn** | Divisão de dados, treino de modelos (Linear, Ridge, Lasso) e computação de métricas ($R^2$ e RMSE). |
| **joblib** | Serialização e armazenamento dinâmico dos dicionários de dados e pipelines de treino. |
| **Streamlit** | Framework para UI interativa com suporte a estados de sessão (`session_state`) e componentes nativos de mapas. |

### 2.3 Comparação com Trabalhos Semelhantes

| Sistema | Abordagem | Escala | Funcionalidades de Interface |
|---|---|---|---|
| **Zillow "Zestimate"** | Redes Neuronais Profundas | Milhões de Imóveis (EUA) | Estimativa em mapa e valorização histórica. |
| **Idealista** | Índices baseados em transações | Nacional (Portugal) | Relatórios estáticos e filtros comerciais. |
| **Este Projeto** | **Regressão com Regularização** | Scraping focado (OLX) | **Mapas em tempo real, gestão de regiões e download CSV**. |

---

## 3. Requisitos

### 3.1 Requisitos Funcionais

| ID | Requisito | Descrição | Prioridade |
|---|---|---|---|
| **RF01** | Segregação de Negócio | Permitir alternar entre previsões de arrendamento mensal ou compra total. | Alta |
| **RF02** | Extração Inteligente | Processar múltiplas colunas do CSV para inferir o preço e área válidos. | Alta |
| **RF03** | Seleção Automática | Avaliar Linear, Ridge e Lasso, salvando autonomamente o algoritmo de maior performance. | Alta |
| **RF04** | Engenharia de Features | Computar a taxa de área útil por divisão habitacional (`area_por_quarto`). | Alta |
| **RF05** | Georreferenciação | Plotar um marcador dinâmico em mapa de acordo com a região selecionada na inferência. | Média |
| **RF06** | Gestão de Regiões | Oferecer a capacidade de ocultar/restaurar distritos específicos da lista de opções da UI. | Média |
| **RF07** | Histórico Local | Registar as simulações efetuadas numa tabela temporária durante a sessão. | Média |
| **RF08** | Exportação de Dados | Permitir o download do histórico gerado no formato universal `.csv`. | Média |

### 3.2 Requisitos Não Funcionais

| ID | Requisito | Categoria | Especificação Técnica |
|---|---|---|---|
| **RNF01** | Tolerância a Falhas | O sistema deve injetar dados sintéticos estruturados se a amostragem física for criticamente baixa. | Confiabilidade |
| **RNF02** | Otimização de Memória | O modelo deve ser carregado através de caching decorado (`@st.cache_resource`). | Desempenho |
| **RNF03** | Persistência Integrada | Os ficheiros `.pkl` devem conter o modelo, colunas de treino e histórico estatístico. | Arquitetura |
| **RNF04** | Resiliência de Input | Inputs que resultem em estimativas matemáticas negativas devem ser truncados para $0.0\ €$. | Robustez |

---

## 4. Arquitetura da Solução

### 4.1 Visão Geral

┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MÓDULO DE TREINO                                                            │
│                                                                                                               │
│  olx-pt-2026-6-08.csv   ──► Extração Segregada ──► [Arrendamento] ──► Otimização ──►  modelo_arrendamento.pkl │
│  olx-pt-2026-6-08-2.csv ──► & One-Hot Encoding ──► [Compra/Venda] ──► (Linear/  ──►  modelo_venda.pkl         │
│                                                                        Ridge/Lasso)                           │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MÓDULO DE INFERÊNCIA                                   │
│                                                                                        │
│                       ┌──► Form Input (Área, Quartos, Região)                          │
│   Ficheiros .pkl ────┼──► Filtro Dinâmico e Remoção de Zonas                           │
│                      └──► Renderização de Mapa (Coordenadas PT) & Exportador CSV       │
└────────────────────────────────────────────────────────────────────────────────────────┘
### 4.2 Pipeline Completo de Dados

1. **Recolha:** Ficheiros raw recolhidos do OLX Portugal (`olx-pt-2026-6-08.csv` e `olx-pt-2026-6-08-2.csv`).

2. **Pré-processamento Inteligente:**
   * **Limpeza de Preço:** Procura sequencial nas colunas `data2`, `data1`, `data3` e `data`. Filtra e ignora strings nocivas ("Gratuito", "Troca", "Sob Consulta").
   * **Filtros Heurísticos de Negócio:** Segmentação por limites monetários (Preços $> 5000$ são classificados como Vendas; Preços entre $50$ e $10000$ são classificados como Arrendamento).
   * **Regex de Quartos:** Conversão de formatos variados ("T3", "3 Quartos", "Estúdio", "T0") em floats normalizados.
   * **Extração de Localização:** Limpeza da string removendo ruídos textuais (ex: "Para o topo a...") para isolar o concelho/distrito real.

3. **Treino e Otimização:** * Expansão de colunas fictícias para localizações (*get_dummies*).
   * Divisão clássica 80% treino / 20% teste.
   * Treino concorrente de **LinearRegression**, **Ridge** e **Lasso**.
   * Análise de $R^2$ em tempo real e persistência da vertente vencedora.

### 4.3 Estrutura de Ficheiros Atualizada

projeto_imoveis/
│
├── ML/
│   └── dados de treino/
│       ├── olx-pt-2026-6-08.csv        ← Dataset focado em arrendamentos
│       └── olx-pt-2026-6-08-2.csv      ← Dataset focado em vendas
│
├── testetreino.py                      ← Script de treino concorrente c/ seleção automática
├── teste.py                            ← Interface avançada Streamlit c/ mapas e histórico
│
├── modelo_arrendamento.pkl             ← Pipeline serializado de arrendamentos
├── modelo_venda.pkl                    ← Pipeline serializado de vendas
└── relatorio_final.md                  ← Este relatório unificado
---

## 5. Implementação Principal

### 5.1 Treino Concorrente e Regularização (`testetreino.py`)

O trecho de código abaixo reflete o novo motor de treino adaptativo com seleção de algoritmos:

```python
modelos = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

melhor_r2 = -float('inf')
melhor_modelo = None

print(f"\n--- Métricas de Avaliação ({tipo}) ---")
for nome, mod in modelos.items():
    mod.fit(X_train, y_train)
    preds_test = mod.predict(X_test)
    preds_train = mod.predict(X_train)
    
    r2_t = r2_score(y_train, preds_train)
    r2_v = r2_score(y_test, preds_test)
    rmse_v = np.sqrt(mean_squared_error(y_test, preds_test))
    
    print(f"[{nome}] Treino R²: {r2_t:.2f} | Teste R²: {r2_v:.2f} | Teste RMSE: {rmse_v:.2f}")
    
    if r2_v > melhor_r2:
        melhor_r2 = r2_v
        melhor_modelo = mod
5.2 Lógica de Inferência e Reindexação na UI (teste.py)A interface recompõe o vetor de características dinamicamente para garantir a compatibilidade com o One-Hot Encoding do treino:Python
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

6. Testes e Avaliação

6.1 Resultados do Novo Treino
A inclusão de modelos de regularização reduziu drasticamente o impacto dos ruídos e do desbalanceamento do dataset anterior. Os modelos conseguiram estabilizar as métricas de treino e teste:

Pipeline Arrendamento: O algoritmo vencedor consegue isolar de forma eficaz coeficientes inflacionados em capitais.

Pipeline Venda: Protegido contra grandes distorções provocadas por outliers, prevenindo previsões excessivamente discrepantes ou irreais.

6.2 Comparações e Métricas Estimadas

| Modelo       | R² Treino | R² Teste | Comportamento Observado |
|--------------|-----------|----------------|--------------------------------------------|
| Linear Comum | Alto      | Variável / Instável | Tende a sofrer com alta variância devido ao número de colunas categóricas.                                                           |
| Ridge (L2)   | Equilibrado | Consistente  | Vencedor frequente: suaviza pesos de regiões com poucos anúncios.                                                                         |
| Lasso (L1)   | Seletivo | Estável         | Zera coeficientes de localizações estatisticamente irrelevantes.                                                                            |

7. Melhorias Técnicas e Correção de ErrosFace à primeira versão do projeto, foram eliminadas com sucesso todas as limitações identificadas:Resolução da Anomalia de Zonas Rurais: Na versão anterior, o aumento da área de casas rurais causava quedas no preço estimado devido à mistura com dados de rendas. A separação estrita em dois ficheiros de modelo autónomos (modelo_arrendamento.pkl e modelo_venda.pkl) extinguiu este conflito.Tratamento de Localizações: A variável geográfica foi reintroduzida através de engenharia robusta de strings, mapeando com precisão distritos críticos (Lisboa, Porto, Braga, Coimbra, Faro, Setúbal).Segurança Estatística (Fallback): O script de treino foi programado com resiliência: caso o dataset de scraping apresente menos de 5 amostras válidas após a limpeza, é injetado automaticamente um conjunto simulado expandido com dados coerentes para mitigar erros de execução.

8. Conclusão e Trabalho FuturoO sistema cumpre com rigor os requisitos académicos e funcionais de uma aplicação preditiva de Machine Learning. A transição para modelos baseados em Ridge e Lasso assegurou a estabilidade matemática do software, enquanto a nova interface em Streamlit elevou o patamar de usabilidade com a manipulação de estados, mapas integrados e histórico exportável.Próximos Passos:Alargar o mapeamento de coordenadas geográficas (COORDENADAS_PT) para cobrir todos os municípios de Portugal Continental de forma nativa.Automatizar a execução diária do script de treino (testetreino.py) através de uma rotina cronometrada para autogestão do modelo imobiliário.

9. Referências Bibliográficas
Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (3rd ed.). O'Reilly Media.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). An Introduction to Statistical Learning: with Applications in Python. Springer.

McKinney, W. (2022). Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter (3rd ed.). O'Reilly Media.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Streamlit Inc. (2026). Streamlit API Reference Documentation. Obtido de https://docs.streamlit.io

The pandas development team. (2026). pandas: powerful Python data analysis toolkit. Obtido de https://pandas.pydata.org/docs/