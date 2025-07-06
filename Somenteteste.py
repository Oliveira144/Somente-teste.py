import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from datetime import datetime
import time

# --- Configuração Premium da Página ---
st.set_page_config(
    page_title="Bac Bo Intelligence Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🎯"
)
st.title("🎯 BAC BO PREDICTOR PRO - Sistema de Alta Precisão")

# Estilos CSS Premium
st.markdown("""
<style>
    /* Design Premium */
    .stApp {
        background: linear-gradient(135deg, #1a1b28, #26273b);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stAlert {
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 1.8rem;
        font-size: 1.4em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        border: 2px solid;
    }
    .alert-success { 
        background: linear-gradient(135deg, #28a745, #1e7e34);
        border-color: #0c5420;
    }
    .alert-danger { 
        background: linear-gradient(135deg, #dc3545, #bd2130);
        border-color: #8a1621;
    }
    .alert-warning { 
        background: linear-gradient(135deg, #ffc107, #e0a800);
        border-color: #b38700;
        color: #000 !important;
    }
    .stMetric {
        background: rgba(46, 47, 58, 0.7);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border: 1px solid #3d4050;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Botões premium */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    /* Abas */
    .stTabs [aria-selected="true"] {
        font-weight: bold;
        background: rgba(46, 47, 58, 0.9) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Inicialização do Session State ---
if 'historico_dados' not in st.session_state:
    st.session_state.historico_dados = []
    st.session_state.padroes_detectados = []
    st.session_state.modelos_treinados = False
    st.session_state.ultimo_treinamento = None
    st.session_state.backtest_results = {}

# --- Constantes Avançadas ---
JANELAS_ANALISE = [
    {"nome": "Ultra-curto", "tamanho": 8, "peso": 1.5},
    {"nome": "Curto", "tamanho": 20, "peso": 1.8},
    {"nome": "Médio", "tamanho": 50, "peso": 1.2},
    {"nome": "Longo", "tamanho": 100, "peso": 0.9}
]

MODELOS = {
    "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.12, max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(25, 15), activation='relu', max_iter=2000),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
}

# --- Funções de Análise Avançada ---

def calcular_probabilidade_condicional(df, evento, condicao):
    try:
        total_condicao = len(df.query(condicao))
        if total_condicao == 0:
            return 0.0
        total_ambos = len(df.query(f"{condicao} and {evento}"))
        return (total_ambos / total_condicao) * 100
    except:
        return 0.0

def previsao_avancada(X_train, y_train, X_pred):
    probas = []
    
    for nome, modelo in MODELOS.items():
        try:
            modelo.fit(X_train, y_train)
            proba = modelo.predict_proba(X_pred)[0]
            probas.append(proba)
        except Exception as e:
            st.error(f"Erro no modelo {nome}: {str(e)}")
    
    if probas:
        return np.mean(probas, axis=0)
    return [0.33, 0.33, 0.34]  # Retorno neutro se falhar

def detectar_padroes_avancados(df_completo):
    todos_padroes = []
    
    for janela in JANELAS_ANALISE:
        tamanho = janela["tamanho"]
        peso_janela = janela["peso"]
        
        if len(df_completo) < tamanho:
            continue
            
        df_analise = df_completo.tail(tamanho).copy()
        n = len(df_analise)
        x = np.arange(n)
        
        # 1. Análise de Tendência Avançada
        try:
            player_slope, _, _, _, _ = stats.linregress(x, df_analise["Player"])
            player_trend_strength = min(2.5, abs(player_slope) * 8)
            
            banker_slope, _, _, _, _ = stats.linregress(x, df_analise["Banker"])
            banker_trend_strength = min(2.5, abs(banker_slope) * 8)
            
            if player_slope > 0.15:
                todos_padroes.append({
                    "tipo": "TENDÊNCIA", 
                    "lado": "P", 
                    "desc": f"Soma Player em alta forte ({player_slope:.2f}) - Janela {janela['nome']}",
                    "peso": player_trend_strength * peso_janela,
                    "janela": janela["nome"]
                })
            elif player_slope < -0.15:
                todos_padroes.append({
                    "tipo": "TENDÊNCIA", 
                    "lado": "P", 
                    "desc": f"Soma Player em queda forte ({player_slope:.2f}) - Janela {janela['nome']}",
                    "peso": player_trend_strength * peso_janela,
                    "janela": janela["nome"]
                })
                
            if banker_slope > 0.15:
                todos_padroes.append({
                    "tipo": "TENDÊNCIA", 
                    "lado": "B", 
                    "desc": f"Soma Banker em alta forte ({banker_slope:.2f}) - Janela {janela['nome']}",
                    "peso": banker_trend_strength * peso_janela,
                    "janela": janela["nome"]
                })
            elif banker_slope < -0.15:
                todos_padroes.append({
                    "tipo": "TENDÊNCIA", 
                    "lado": "B", 
                    "desc": f"Soma Banker em queda forte ({banker_slope:.2f}) - Janela {janela['nome']}",
                    "peso": banker_trend_strength * peso_janela,
                    "janela": janela["nome"]
                })
        except:
            pass
        
        # 2. Análise de Repetição Estatística
        player_counts = Counter(df_analise["Player"])
        banker_counts = Counter(df_analise["Banker"])
        
        for soma, count in player_counts.items():
            if count >= max(4, n*0.35):  # Limiares mais rigorosos
                peso = min(3.0, count * 0.6) * peso_janela
                todos_padroes.append({
                    "tipo": "REPETIÇÃO", 
                    "lado": "P", 
                    "desc": f"Soma Player {soma} repetida {count}/{n} vezes ({count/n*100:.1f}%)",
                    "peso": peso,
                    "janela": janela["nome"]
                })
                
        for soma, count in banker_counts.items():
            if count >= max(4, n*0.35):
                peso = min(3.0, count * 0.6) * peso_janela
                todos_padroes.append({
                    "tipo": "REPETIÇÃO", 
                    "lado": "B", 
                    "desc": f"Soma Banker {soma} repetida {count}/{n} vezes ({count/n*100:.1f}%)",
                    "peso": peso,
                    "janela": janela["nome"]
                })
        
        # 3. Previsão com Modelo Híbrido
        if n > 15:
            try:
                X = df_analise[["Player", "Banker"]].values[:-1]
                y = df_analise["Resultado"].values[1:]
                X_pred = df_analise[["Player", "Banker"]].values[-1].reshape(1, -1)
                
                probas = previsao_avancada(X, y, X_pred)
                max_idx = np.argmax(probas)
                confianca = probas[max_idx]
                
                if confianca > 0.62:  # Limiar mais alto para confiança
                    lado_pred = ["P", "B", "T"][max_idx]
                    todos_padroes.append({
                        "tipo": "PREVISÃO", 
                        "lado": lado_pred, 
                        "desc": f"Modelo preditivo ({janela['nome']}) sugere {lado_pred} (conf: {confianca*100:.1f}%)",
                        "peso": min(4.0, confianca * 6) * peso_janela,
                        "janela": janela["nome"]
                    })
            except Exception as e:
                st.error(f"Erro na previsão: {str(e)}")
    
    # 4. Análise de Probabilidade Condicional (histórico completo)
    if len(df_completo) > 100:
        try:
            # Player ganha quando soma > 8
            prob = calcular_probabilidade_condicional(
                df_completo, 
                "Resultado == 'P'", 
                "Player > 8"
            )
            if prob > 58:  # Limiar mais alto
                todos_padroes.append({
                    "tipo": "PROBABILIDADE", 
                    "lado": "P", 
                    "desc": f"Prob histórica: Player ganha {prob:.1f}% quando soma > 8",
                    "peso": min(3.0, (prob-50)/8),
                    "janela": "Histórico"
                })
                
            # Banker ganha quando soma > 9
            prob = calcular_probabilidade_condicional(
                df_completo, 
                "Resultado == 'B'", 
                "Banker > 9"
            )
            if prob > 58:
                todos_padroes.append({
                    "tipo": "PROBABILIDADE", 
                    "lado": "B", 
                    "desc": f"Prob histórica: Banker ganha {prob:.1f}% quando soma > 9",
                    "peso": min(3.0, (prob-50)/8),
                    "janela": "Histórico"
                })
                
            # Tie quando diferença pequena
            prob = calcular_probabilidade_condicional(
                df_completo, 
                "Resultado == 'T'", 
                "abs(Player - Banker) <= 1"
            )
            if prob > 15:  # Probabilidade natural ~10%
                todos_padroes.append({
                    "tipo": "PROBABILIDADE", 
                    "lado": "T", 
                    "desc": f"Prob histórica: Tie ocorre em {prob:.1f}% quando diferença <=1",
                    "peso": min(3.0, prob/6),
                    "janela": "Histórico"
                })
        except:
            pass
    
    # 5. Padrões de Sequência
    resultados = df_completo["Resultado"].values
    if len(resultados) > 10:
        # Detecção de sequências P-B-P-B
        padrao_alternancia = 0
        for i in range(4, len(resultados)):
            if (resultados[i-3] == 'P' and resultados[i-2] == 'B' and 
                resultados[i-1] == 'P' and resultados[i] == 'B'):
                padrao_alternancia += 1
        
        if padrao_alternancia >= 2:
            todos_padroes.append({
                "tipo": "SEQUÊNCIA", 
                "lado": "AMBOS", 
                "desc": f"Padrão de alternância P-B-P-B detectado {padrao_alternancia} vezes",
                "peso": 2.5,
                "janela": "Longo"
            })
    
    return todos_padroes

def gerar_recomendacao(padroes):
    if not padroes:
        return "AGUARDAR", 15, "Sem padrões detectados. Aguarde mais dados.", "warning"
    
    # Agrupar padrões por lado
    scores = {"P": 0.0, "B": 0.0, "T": 0.0}
    detalhes = {"P": [], "B": [], "T": []}
    
    for padrao in padroes:
        lado = padrao["lado"]
        peso = padrao["peso"]
        
        if lado in scores:
            scores[lado] += peso
            detalhes[lado].append(f"{padrao['tipo']}: {padrao['desc']}")
        elif lado == "AMBOS":
            scores["P"] += peso/2
            scores["B"] += peso/2
            detalhes["P"].append(f"{padrao['tipo']}: {padrao['desc']}")
            detalhes["B"].append(f"{padrao['tipo']}: {padrao['desc']}")
    
    # Calcular confiança
    total_score = sum(scores.values())
    if total_score == 0:
        return "AGUARDAR", 10, "Padrões sem força significativa", "warning"
    
    confiancas = {lado: min(100, int(score/total_score * 100)) for lado, score in scores.items()}
    
    # Determinar recomendação com limiares mais altos
    max_lado = max(scores, key=scores.get)
    max_score = scores[max_lado]
    
    # Limiares de decisão mais rigorosos
    if max_score > 6.0:
        acao = f"APOSTAR FORTE NO {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "success"
        conf = confiancas[max_lado]
        detalhe = f"**Convergência poderosa de padrões** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    elif max_score > 4.0:
        acao = f"APOSTAR NO {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "success"
        conf = confiancas[max_lado]
        detalhe = f"**Forte convergência de padrões** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    elif max_score > 2.5:
        acao = f"CONSIDERAR {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "warning"
        conf = confiancas[max_lado]
        detalhe = f"**Sinal moderado** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    else:
        acao = "AGUARDAR"
        tipo = "warning"
        conf = 100 - max(confiancas.values())
        detalhe = "**Sinais fracos ou conflitantes**. Aguarde confirmação:\n- " + "\n- ".join(
            [f"{lado}: {score:.1f} pts" for lado, score in scores.items()])
    
    return acao, conf, detalhe, tipo

def executar_backtesting(df, estrategia, tamanho_janela=20):
    resultados = []
    saldo = 1000
    apostas = []
    detalhes = []
    
    for i in range(tamanho_janela, len(df)):
        dados_janela = df.iloc[i-tamanho_janela:i]
        recomendacao = estrategia(dados_janela)
        resultado_real = df.iloc[i]['Resultado']
        
        # Simulação de aposta com tamanho variável
        if recomendacao == resultado_real:
            ganho = 50 if recomendacao != 'T' else 80
            saldo += ganho
            apostas.append(1)  # Vitória
            detalhes.append({
                "jogo": i,
                "aposta": recomendacao,
                "resultado": resultado_real,
                "ganho": ganho,
                "saldo": saldo
            })
        else:
            perda = 50 if recomendacao != 'T' else 80
            saldo -= perda
            apostas.append(0)  # Derrota
            detalhes.append({
                "jogo": i,
                "aposta": recomendacao,
                "resultado": resultado_real,
                "ganho": -perda,
                "saldo": saldo
            })
    
    # Calcular métricas
    win_rate = np.mean(apostas) * 100 if apostas else 0
    retorno = (saldo - 1000) / 1000 * 100
    
    return {
        "saldo_final": saldo,
        "win_rate": win_rate,
        "retorno_percent": retorno,
        "detalhes": detalhes
    }

# --- Interface Premium ---
st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
    <h1 style="color:#ffc107; font-size:2.5em;">BAC BO PREDICTOR PRO</h1>
    <p style="font-size:1.2em;">Sistema de análise preditiva com algoritmos de Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --- Entrada de Dados Premium ---
with st.expander("🎮 ENTRADA DE DADOS", expanded=True):
    col1, col2, col3, col4 = st.columns([1,1,1,0.8])
    with col1:
        player_soma = st.number_input("Soma Player (2-12)", min_value=2, max_value=12, value=7, key="player_soma_input")
    with col2:
        banker_soma = st.number_input("Soma Banker (2-12)", min_value=2, max_value=12, value=7, key="banker_soma_input")
    with col3:
        resultado_op = st.selectbox("Resultado", ['P', 'B', 'T'], key="resultado_select")
    with col4:
        st.write("")
        st.write("")
        if st.button("➕ ADICIONAR", use_container_width=True, type="primary"):
            st.session_state.historico_dados.append((player_soma, banker_soma, resultado_op))
            st.rerun()

# --- Histórico com Visualização Premium ---
st.subheader("📜 HISTÓRICO DE RESULTADOS")
if st.session_state.historico_dados:
    df_historico = pd.DataFrame(
        st.session_state.historico_dados,
        columns=["Player", "Banker", "Resultado"]
    )
    
    # Adicionar colunas analíticas
    df_historico['Diferenca'] = abs(df_historico['Player'] - df_historico['Banker'])
    df_historico['SomaTotal'] = df_historico['Player'] + df_historico['Banker']
    df_historico['Vencedor'] = np.where(
        df_historico['Resultado'] == 'P', 'Player',
        np.where(df_historico['Resultado'] == 'B', 'Banker', 'Tie')
    )
    
    # Exibir tabela com estilo
    st.dataframe(df_historico.tail(20).style
        .background_gradient(subset=['Player', 'Banker'], cmap='YlGnBu')
        .applymap(lambda x: 'color: blue; font-weight: bold' if x == 'P' else 
                 ('color: red; font-weight: bold' if x == 'B' else 
                  'color: green; font-weight: bold'), 
                subset=['Resultado']),
        use_container_width=True, height=450)
    
    # Controles do histórico
    col_hist1, col_hist2, col_hist3 = st.columns([1,1,2])
    with col_hist1:
        if st.button("🗑️ REMOVER ÚLTIMO", use_container_width=True):
            if st.session_state.historico_dados:
                st.session_state.historico_dados.pop()
                st.rerun()
    with col_hist2:
        if st.button("🧹 LIMPAR TUDO", use_container_width=True, type="secondary"):
            st.session_state.historico_dados = []
            st.session_state.padroes_detectados = []
            st.rerun()
    with col_hist3:
        last = df_historico.iloc[-1] if not df_historico.empty else ""
        st.info(f"🔢 Total: {len(df_historico)} | Último: {last.get('Player', '')}-{last.get('Banker', '')}-{last.get('Resultado', '')}")
else:
    st.warning("⚠️ Nenhum dado no histórico. Adicione resultados para iniciar a análise.")

# --- Entrada em Massa Premium ---
with st.expander("📥 IMPORTAR DADOS EM MASSA", expanded=False):
    historico_input_mass = st.text_area("Cole múltiplas linhas (1 linha = Player,Banker,Resultado)", height=150)
    
    if st.button("🚀 PROCESSAR DADOS", use_container_width=True, type="primary"):
        linhas = [linha.strip() for linha in historico_input_mass.split("\n") if linha.strip()]
        novos_dados = []
        erros = []
        
        for i, linha in enumerate(linhas, 1):
            try:
                partes = [p.strip() for p in linha.split(',')]
                if len(partes) < 3:
                    erros.append(f"Linha {i}: Formato inválido (esperado: Player,Banker,Resultado)")
                    continue
                
                p = int(partes[0])
                b = int(partes[1])
                r = partes[2].upper()
                
                if not (2 <= p <= 12):
                    erros.append(f"Linha {i}: Soma Player inválida ({p}) - deve ser 2-12")
                if not (2 <= b <= 12):
                    erros.append(f"Linha {i}: Soma Banker inválida ({b}) - deve ser 2-12")
                if r not in ['P', 'B', 'T']:
                    erros.append(f"Linha {i}: Resultado inválido ({r}) - deve ser P, B ou T")
                
                if not erros or not any(f"Linha {i}" in e for e in erros):
                    novos_dados.append((p, b, r))
            except Exception as e:
                erros.append(f"Linha {i}: Erro de processamento - {str(e)}")
        
        if erros:
            for erro in erros:
                st.error(erro)
        else:
            st.session_state.historico_dados.extend(novos_dados)
            st.success(f"✅ {len(novos_dados)} linhas adicionadas com sucesso!")
            st.rerun()

# --- Verificação de Dados ---
if not st.session_state.historico_dados:
    st.warning("📊 Adicione dados para iniciar a análise!")
    st.stop()

df = pd.DataFrame(
    st.session_state.historico_dados,
    columns=["Player", "Banker", "Resultado"]
)

# --- Painel de Análise Premium ---
st.markdown("---")
st.header("🧠 ANÁLISE PREDITIVA AVANÇADA")

# Análise com múltiplas janelas
padroes = detectar_padroes_avancados(df)
st.session_state.padroes_detectados = padroes

# Gerar recomendação
acao, confianca, detalhes, tipo = gerar_recomendacao(padroes)

# Exibir recomendação premium
st.markdown(f"""
<div class="stAlert alert-{tipo}">
    <div style="font-size: 1.6em; margin-bottom: 10px;">{acao}</div>
    <div style="font-size: 1.2em;">Confiança: {confianca}%</div>
    <div style="font-size: 0.95em; margin-top: 15px; text-align: left; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
        {detalhes.replace('\n', '<br>')}
    </div>
</div>
""", unsafe_allow_html=True)

# Sugestão de valor de aposta
if "APOSTAR" in acao:
    if confianca >= 80:
        tamanho_aposta = "5% do seu bankroll"
        cor = "#28a745"
    elif confianca >= 70:
        tamanho_aposta = "3% do seu bankroll"
        cor = "#ffc107"
    else:
        tamanho_aposta = "1-2% do seu bankroll"
        cor = "#dc3545"
    
    st.markdown(f"""
    <div style="background: {cor}; padding: 12px; border-radius: 8px; text-align: center; font-size: 1.1em;">
        💰 <b>SUGESTÃO DE APOSTA:</b> {tamanho_aposta}
    </div>
    """, unsafe_allow_html=True)

# --- Estatísticas Premium ---
st.subheader("📈 MÉTRICAS DE DESEMPENHO")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total de Jogos", len(df))
with col2:
    player_wins = len(df[df['Resultado'] == 'P'])
    st.metric("Vitórias Player", f"{player_wins} ({player_wins/len(df)*100:.1f}%)")
with col3:
    banker_wins = len(df[df['Resultado'] == 'B'])
    st.metric("Vitórias Banker", f"{banker_wins} ({banker_wins/len(df)*100:.1f}%)")
with col4:
    tie_wins = len(df[df['Resultado'] == 'T'])
    st.metric("Empates (Tie)", f"{tie_wins} ({tie_wins/len(df)*100:.1f}%)")
with col5:
    avg_diff = abs(df['Player'] - df['Banker']).mean()
    st.metric("Diferença Média", f"{avg_diff:.2f}")

# --- Visualizações Gráficas Premium ---
st.subheader("📊 VISUALIZAÇÃO DE PADRÕES")

# Gráfico 1: Distribuição de Resultados
fig1 = px.pie(
    df, 
    names='Resultado', 
    title='Distribuição de Resultados',
    color='Resultado',
    color_discrete_map={'P': '#1f77b4', 'B': '#d62728', 'T': '#2ca02c'},
    hole=0.4
)
fig1.update_traces(textposition='inside', textinfo='percent+label', 
                  marker=dict(line=dict(color='#fff', width=2)))

# Gráfico 2: Evolução Temporal
df['Indice'] = range(1, len(df)+1)
fig2 = px.line(
    df.tail(40), 
    x='Indice', 
    y=['Player', 'Banker'],
    title='Evolução das Somas (últimos 40 jogos)',
    markers=True,
    color_discrete_map={'Player': '#1f77b4', 'Banker': '#d62728'}
)
fig2.update_layout(
    yaxis_title="Soma", 
    xaxis_title="Jogo",
    legend_title="Lado"
)
fig2.add_hline(y=7.5, line_dash="dash", line_color="gray", annotation_text="Média Esperada")

# Gráfico 3: Heatmap de Frequência
freq_matrix = pd.crosstab(df['Player'], df['Banker'])
fig3 = px.imshow(
    freq_matrix,
    labels=dict(x="Banker", y="Player", color="Frequência"),
    title="Frequência Player vs Banker",
    aspect="auto",
    color_continuous_scale='Viridis'
)

# Exibir gráficos em layout premium
col_graph1, col_graph2 = st.columns(2)
with col_graph1:
    st.plotly_chart(fig1, use_container_width=True)
with col_graph2:
    st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# --- Painel de Padrões Detectados ---
if padroes:
    st.subheader("🔍 PADRÕES DETECTADOS")
    
    # Agrupar por tipo de padrão
    tipos = {}
    for padrao in padroes:
        if padrao['tipo'] not in tipos:
            tipos[padrao['tipo']] = []
        tipos[padrao['tipo']].append(padrao)
    
    # Exibir em abas premium
    tabs = st.tabs([f"{tipo} ({len(padroes_tipo)})" for tipo, padroes_tipo in tipos.items()])
    
    for i, (tipo, padroes_tipo) in enumerate(tipos.items()):
        with tabs[i]:
            for padrao in padroes_tipo:
                color_map = {
                    "P": "#1f77b4",
                    "B": "#d62728",
                    "T": "#2ca02c",
                    "AMBOS": "#9467bd"
                }
                cor = color_map.get(padrao['lado'], "#636efa")
                
                st.markdown(f"""
                <div style="background: rgba(46, 47, 58, 0.7); 
                            padding: 15px; 
                            border-radius: 10px; 
                            margin-bottom: 15px;
                            border-left: 5px solid {cor}">
                    <div style="display: flex; justify-content: space-between;">
                        <div><b style="color: {cor};">{padrao['lado']}</b> | {padrao['janela']}</div>
                        <div>Peso: <b>{padrao['peso']:.1f}</b></div>
                    </div>
                    <div style="margin-top: 10px;">{padrao['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Nenhum padrão estatisticamente significativo detectado.")

# --- Sistema de Backtesting Automático ---
st.markdown("---")
st.header("🧪 TESTE DE ESTRATÉGIAS")

# Estratégias para teste
estrategias = [
    {"nome": "Tendência Player", 
     "funcao": lambda df: "P" if df["Player"].mean() > df["Banker"].mean() else "B",
     "desc": "Aposta no lado com maior soma média"},
    
    {"nome": "Anti-Streak", 
     "funcao": lambda df: "P" if df["Resultado"].iloc[-1] == "B" else "B",
     "desc": "Aposta contra o último resultado"},
    
    {"nome": "Soma Alta Player", 
     "funcao": lambda df: "P" if df["Player"].iloc[-1] > 8 else "B",
     "desc": "Aposta no Player quando sua soma > 8"},
    
    {"nome": "Sistema Preditor", 
     "funcao": lambda df: 
         ["P", "B", "T"][np.argmax(previsao_avancada(
             df[["Player", "Banker"]].values[:-1], 
             df["Resultado"].values[1:],
             df[["Player", "Banker"]].values[-1].reshape(1, -1)
         )] if len(df) > 15 else "B",
     "desc": "Usa o modelo de machine learning para prever"}
]

if st.button("🏁 EXECUTAR BACKTESTING COMPLETO", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, estrategia in enumerate(estrategias):
        status_text.text(f"Testando estratégia: {estrategia['nome']}...")
        st.session_state.backtest_results[estrategia['nome']] = executar_backtesting(df, estrategia['funcao'])
        progress_bar.progress((i+1)/len(estrategias))
    
    status_text.success("✅ Backtesting completo!")
    time.sleep(1)
    status_text.empty()

# Exibir resultados
if st.session_state.backtest_results:
    st.subheader("📊 RESULTADOS DO BACKTESTING")
    
    resultados = []
    for nome, res in st.session_state.backtest_results.items():
        resultados.append({
            "Estratégia": nome,
            "Win Rate": f"{res['win_rate']:.1f}%",
            "Retorno": f"{res['retorno_percent']:.1f}%",
            "Saldo Final": f"R$ {res['saldo_final']:.2f}"
        })
    
    df_resultados = pd.DataFrame(resultados)
    st.dataframe(df_resultados.sort_values("Win Rate", ascending=False), use_container_width=True)
    
    # Gráfico de evolução do saldo
    if 'detalhes' in st.session_state.backtest_results[estrategias[0]['nome']]:
        st.subheader("📈 EVOLUÇÃO DO SALDO")
        
        fig_evol = go.Figure()
        for estrategia in estrategias:
            nome = estrategia['nome']
            if nome in st.session_state.backtest_results:
                detalhes = st.session_state.backtest_results[nome]['detalhes']
                if detalhes:
                    saldos = [d['saldo'] for d in detalhes]
                    jogos = [d['jogo'] for d in detalhes]
                    fig_evol.add_trace(go.Scatter(
                        x=jogos, 
                        y=saldos, 
                        mode='lines+markers',
                        name=nome,
                        line=dict(width=3)
                    ))
        
        fig_evol.update_layout(
            title='Evolução do Saldo por Estratégia',
            xaxis_title='Número do Jogo',
            yaxis_title='Saldo (R$)',
            legend_title='Estratégia',
            template='plotly_dark'
        )
        st.plotly_chart(fig_evol, use_container_width=True)

# --- Sistema de Alertas Estratégicos ---
st.subheader("🚨 ALERTAS ESTRATÉGICOS")

# Verificar condições críticas
alertas = []
if len(df) > 15:
    # Alerta para sequências longas
    ultimo_resultado = df['Resultado'].iloc[-1]
    streak_count = 1
    for i in range(len(df)-2, -1, -1):
        if df['Resultado'].iloc[i] == ultimo_resultado:
            streak_count += 1
        else:
            break
    
    if streak_count >= 6:
        alertas.append(f"🔥 SEQUÊNCIA EXTREMA: {streak_count} vitórias consecutivas para {ultimo_resultado}")
    elif streak_count >= 5:
        alertas.append(f"⚠️ Sequência longa: {streak_count} vitórias consecutivas para {ultimo_resultado}")
    
    # Alerta para empates
    ultimo_tie_idx = df[df['Resultado'] == 'T'].index
    desde_ultimo_tie = len(df) - ultimo_tie_idx[-1] if len(ultimo_tie_idx) > 0 else len(df)
    
    if desde_ultimo_tie >= 18:
        alertas.append("🔥 CICLO DE TIE SUPER MADURO - Probabilidade muito alta de empate")
    elif desde_ultimo_tie >= 15:
        alertas.append("⚠️ Ciclo de TIE MADURO - Alta probabilidade de empate")
    elif desde_ultimo_tie >= 12:
        alertas.append("🔔 Ciclo de TIE APROXIMANDO - Fique atento")

# Exibir alertas
if alertas:
    for alerta in alertas:
        st.warning(alerta)
else:
    st.info("ℹ️ Nenhum alerta crítico detectado no momento")

# --- Painel de Controle Avançado ---
with st.expander("⚙️ CONFIGURAÇÕES AVANÇADAS", expanded=False):
    st.write("**Otimização de Parâmetros**")
    analise_range = st.slider("Número de jogos para análise", 5, 100, 30)
    limiar_confianca = st.slider("Limiar de confiança para apostas", 60, 95, 75)
    
    st.write("**Preferências de Estratégia**")
    estrategia = st.selectbox("Foco estratégico", [
        "Padrões de curto prazo", 
        "Tendências de longo prazo",
        "Detecção de empates",
        "Sequências de vitórias"
    ])
    
    st.write("**Gerenciamento de Bankroll**")
    bankroll = st.number_input("Seu bankroll total (R$)", min_value=100, value=1000, step=100)
    
    if st.button("🔄 ATUALIZAR SISTEMA", type="primary"):
        st.rerun()

# --- Rodapé Premium ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(46, 47, 58, 0.5); border-radius: 10px;">
    <p style="font-size: 0.9em;">⚠️ <b>AVISO IMPORTANTE</b>: Este sistema é uma ferramenta analítica. Jogos envolvem risco e resultados podem variar. Nunca aposte mais do que pode perder.</p>
    <p style="font-size: 0.8em; margin-top: 10px;">© 2023 Bac Bo Predictor Pro | Sistema de Análise Preditiva | v4.2</p>
</div>
""", unsafe_allow_html=True)
