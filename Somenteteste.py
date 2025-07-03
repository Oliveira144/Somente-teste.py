import streamlit as st
import math
import time
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# Probabilidades teóricas dos dados (soma de 2 dados)
DICE_PROBABILITIES = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 7: 6/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
}

# Inicializar o estado da sessão
if 'results' not in st.session_state:
    st.session_state.results = []

if 'current_stats' not in st.session_state:
    st.session_state.current_stats = {
        'player': 0,
        'banker': 0,
        'tie': 0,
        'totalGames': 0
    }

if 'advanced_analysis' not in st.session_state:
    st.session_state.advanced_analysis = {
        'patterns': {},
        'predictions': [],
        'confidence': 0,
        'volatility': 0,
        'momentum': 0,
        'cyclicalTrends': {},
        'riskLevel': 'LOW'
    }

# Funções auxiliares
def calculate_surprise(pScore, bScore):
    pProb = DICE_PROBABILITIES.get(pScore, 0)
    bProb = DICE_PROBABILITIES.get(bScore, 0)
    combinedProb = pProb * bProb
    return round((1 - combinedProb) * 100)

def add_result(player_score, banker_score):
    try:
        pScore = int(player_score)
        bScore = int(banker_score)
    except:
        st.error("Por favor, insira números válidos.")
        return

    if pScore < 2 or pScore > 12 or bScore < 2 or bScore > 12:
        st.error("As pontuações devem estar entre 2 e 12.")
        return

    if pScore > bScore:
        outcome = 'PLAYER'
        color = 'blue'
    elif bScore > pScore:
        outcome = 'BANKER'
        color = 'red'
    else:
        outcome = 'TIE'
        color = 'green'

    new_result = {
        'id': time.time(),
        'player': pScore,
        'banker': bScore,
        'outcome': outcome,
        'color': color,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'playerProb': DICE_PROBABILITIES[pScore],
        'bankerProb': DICE_PROBABILITIES[bScore],
        'surprise': calculate_surprise(pScore, bScore),
        'gameNumber': len(st.session_state.results) + 1
    }

    # Adiciona e mantém no máximo 100 resultados
    st.session_state.results = [new_result] + st.session_state.results[:99]
    perform_advanced_analysis()

def calculate_basic_stats():
    stats = {
        'player': 0,
        'banker': 0,
        'tie': 0,
        'totalGames': len(st.session_state.results)
    }

    for result in st.session_state.results:
        stats[result['outcome'].lower()] += 1

    return stats

def analyze_streaks():
    results = st.session_state.results
    if not results:
        return {}

    current_streak = {'type': None, 'count': 0}
    streaks = {'player': [], 'banker': [], 'tie': []}

    for result in results:
        outcome = result['outcome'].lower()
        if current_streak['type'] == outcome:
            current_streak['count'] += 1
        else:
            if current_streak['type'] and current_streak['count'] > 1:
                streaks[current_streak['type']].append(current_streak['count'])
            current_streak = {'type': outcome, 'count': 1}

    if current_streak['type'] and current_streak['count'] > 1:
        streaks[current_streak['type']].append(current_streak['count'])

    max_streaks = {
        'player': max(streaks['player']) if streaks['player'] else 0,
        'banker': max(streaks['banker']) if streaks['banker'] else 0,
        'tie': max(streaks['tie']) if streaks['tie'] else 0
    }

    avg_streaks = {
        'player': sum(streaks['player'])/len(streaks['player']) if streaks['player'] else 0,
        'banker': sum(streaks['banker'])/len(streaks['banker']) if streaks['banker'] else 0,
        'tie': sum(streaks['tie'])/len(streaks['tie']) if streaks['tie'] else 0
    }

    return {
        'maxStreaks': max_streaks,
        'avgStreaks': avg_streaks,
        'currentStreak': current_streak
    }

def analyze_alternations():
    results = st.session_state.results
    if len(results) < 3:
        return {'rate': 0, 'pattern': 'NONE'}

    alternations = 0
    consecutives = 0

    for i in range(1, len(results)):
        if results[i]['outcome'] != results[i-1]['outcome']:
            alternations += 1
        else:
            consecutives += 1

    alternation_rate = alternations / (len(results) - 1)
    pattern = 'HIGH_ALT' if alternation_rate > 0.6 else 'LOW_ALT' if alternation_rate < 0.4 else 'BALANCED'

    return {'rate': alternation_rate, 'pattern': pattern, 'alternations': alternations, 'consecutives': consecutives}

def analyze_hot_cold_numbers():
    number_freq = {'player': defaultdict(int), 'banker': defaultdict(int)}
    
    for result in st.session_state.results:
        number_freq['player'][result['player']] += 1
        number_freq['banker'][result['banker']] += 1

    def get_hot_cold(freq):
        entries = [{'num': int(k), 'count': v} for k, v in freq.items()]
        entries.sort(key=lambda x: x['count'], reverse=True)
        return {
            'hot': entries[:3],
            'cold': entries[-3:][::-1]
        }

    return {
        'player': get_hot_cold(number_freq['player']),
        'banker': get_hot_cold(number_freq['banker'])
    }

def analyze_distribution():
    results = st.session_state.results
    n = len(results)
    if n == 0:
        return {'deviations': {}, 'expected': {}}
    
    expected_player = n * 0.486
    expected_banker = n * 0.486
    expected_tie = n * 0.028

    stats = st.session_state.current_stats
    deviations = {
        'player': abs(stats['player'] - expected_player) / expected_player,
        'banker': abs(stats['banker'] - expected_banker) / expected_banker,
        'tie': abs(stats['tie'] - expected_tie) / (expected_tie or 1)
    }

    return {'deviations': deviations, 'expected': {'player': expected_player, 'banker': expected_banker, 'tie': expected_tie}}

def analyze_correlations():
    results = st.session_state.results
    if len(results) < 10:
        return {}
    
    player_corr = 0
    banker_corr = 0
    n = min(len(results) - 1, 20)

    for i in range(1, n+1):
        curr = results[i]
        prev = results[i-1]
        player_corr += curr['player'] * prev['player']
        banker_corr += curr['banker'] * prev['banker']

    return {
        'playerNumberCorrelation': player_corr / n,
        'bankerNumberCorrelation': banker_corr / n,
        'outcomeCorrelation': analyze_outcome_correlation()
    }

def analyze_outcome_correlation():
    results = st.session_state.results
    if len(results) < 6:
        return 0

    matches = 0
    n = min(len(results) - 3, 12)

    for i in range(3, 3+n):
        if results[i]['outcome'] == results[i-3]['outcome']:
            matches += 1

    return matches / n

def analyze_sequences():
    results = st.session_state.results
    sequences = defaultdict(int)
    
    for length in range(2, 5):
        for i in range(len(results) - length + 1):
            seq = '-'.join(r['outcome'] for r in results[i:i+length])
            sequences[seq] += 1

    top_sequences = sorted([(seq, count) for seq, count in sequences.items()], 
                          key=lambda x: x[1], reverse=True)[:5]
    
    top_sequences = [{
        'sequence': seq,
        'count': count,
        'probability': count / len(results)
    } for seq, count in top_sequences]

    return {'sequences': sequences, 'topSequences': top_sequences}

def analyze_cyclical_trends():
    results = st.session_state.results
    if len(results) < 12:
        return {}

    cycles = [3, 5, 7, 10]
    cyclical_data = {}

    for cycle in cycles:
        buckets = [{'player': 0, 'banker': 0, 'tie': 0} for _ in range(cycle)]
        
        for i, result in enumerate(results):
            bucket = i % cycle
            outcome = result['outcome'].lower()
            buckets[bucket][outcome] += 1

        dominant_phases = []
        for phase, bucket in enumerate(buckets):
            total = bucket['player'] + bucket['banker'] + bucket['tie']
            if total == 0:
                dominant_phases.append({'phase': phase, 'dominant': 'NONE', 'strength': 0})
                continue
            
            max_val = max(bucket['player'], bucket['banker'], bucket['tie'])
            if max_val == bucket['player']:
                dominant = 'PLAYER'
            elif max_val == bucket['banker']:
                dominant = 'BANKER'
            else:
                dominant = 'TIE'
                
            dominant_phases.append({'phase': phase, 'dominant': dominant, 'strength': max_val / total})

        cyclical_data[f'cycle{cycle}'] = {
            'buckets': buckets,
            'dominantPhases': dominant_phases,
            'currentPhase': len(results) % cycle,
            'predictedNext': dominant_phases[len(results) % cycle] if dominant_phases else None
        }

    return cyclical_data

def calculate_volatility():
    results = st.session_state.results
    if len(results) < 10:
        return 0

    recent = results[:20]
    changes = 0
    total_surprise = 0

    for i in range(1, len(recent)):
        if recent[i]['outcome'] != recent[i-1]['outcome']:
            changes += 1
        total_surprise += recent[i]['surprise']

    change_rate = changes / (len(recent) - 1)
    avg_surprise = total_surprise / len(recent) if recent else 0
    volatility = min(100, (change_rate * 50) + (avg_surprise * 0.5))
    
    return round(volatility)

def calculate_momentum():
    results = st.session_state.results
    if len(results) < 8:
        return 0

    recent = results[:8]
    weights = [0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
    
    player_momentum = 0
    banker_momentum = 0

    for i, result in enumerate(recent):
        weight = weights[i]
        if result['outcome'] == 'PLAYER':
            player_momentum += weight
        elif result['outcome'] == 'BANKER':
            banker_momentum += weight

    strength = abs(player_momentum - banker_momentum)
    direction = 'PLAYER' if player_momentum > banker_momentum else 'BANKER'

    return {
        'direction': direction,
        'strength': strength,
        'playerMomentum': player_momentum,
        'bankerMomentum': banker_momentum
    }

def generate_predictions():
    predictions = []
    
    # Algoritmo 1: Análise de Reversão
    predictions.append(generate_reversion_prediction())
    
    # Algoritmo 2: Análise de Momentum
    predictions.append(generate_momentum_prediction())
    
    # Algoritmo 3: Análise Cíclica
    predictions.append(generate_cyclical_prediction())
    
    # Algoritmo 4: Análise de Distribuição
    predictions.append(generate_distribution_prediction())
    
    # Algoritmo 5: Análise de Padrões
    predictions.append(generate_pattern_prediction())
    
    return [p for p in predictions if p['confidence'] > 30]

def generate_reversion_prediction():
    results = st.session_state.results
    if len(results) < 5:
        return {'type': 'WAIT', 'confidence': 0, 'reason': 'Dados insuficientes', 'algorithm': 'REVERSION'}

    recent = results[:5]
    last_outcome = recent[0]['outcome']
    same_count = sum(1 for r in recent if r['outcome'] == last_outcome)

    if same_count >= 4:
        opposite = 'BANKER' if last_outcome == 'PLAYER' else 'PLAYER'
        confidence = min(85, 45 + (same_count * 10))
        return {
            'type': opposite,
            'confidence': confidence,
            'reason': f'Reversão após {same_count} {last_outcome}s consecutivos',
            'algorithm': 'REVERSION'
        }

    if same_count >= 3:
        opposite = 'BANKER' if last_outcome == 'PLAYER' else 'PLAYER'
        return {
            'type': opposite,
            'confidence': 60,
            'reason': f'Provável reversão após {same_count} {last_outcome}s',
            'algorithm': 'REVERSION'
        }

    return {'type': 'WAIT', 'confidence': 25, 'reason': 'Sem padrão de reversão claro', 'algorithm': 'REVERSION'}

def generate_momentum_prediction():
    momentum = st.session_state.advanced_analysis.get('momentum', {})
    if not momentum or not isinstance(momentum, dict):
        return {'type': 'WAIT', 'confidence': 0, 'reason': 'Momentum indisponível', 'algorithm': 'MOMENTUM'}
    
    if momentum.get('strength', 0) > 0.3:
        return {
            'type': momentum['direction'],
            'confidence': min(75, 40 + (momentum['strength'] * 100)),
            'reason': f'Momentum forte para {momentum["direction"]}',
            'algorithm': 'MOMENTUM'
        }

    return {
        'type': 'BALANCED',
        'confidence': 35,
        'reason': 'Momentum equilibrado',
        'algorithm': 'MOMENTUM'
    }

def generate_cyclical_prediction():
    cyclical_trends = st.session_state.advanced_analysis.get('cyclicalTrends', {})
    cycle5 = cyclical_trends.get('cycle5', {})
    prediction = cycle5.get('predictedNext', {})
    
    if not prediction or prediction.get('strength', 0) <= 0.4:
        return {
            'type': 'RANDOM',
            'confidence': 30,
            'reason': 'Sem padrão cíclico claro',
            'algorithm': 'CYCLICAL'
        }
    
    return {
        'type': prediction['dominant'],
        'confidence': min(70, 30 + (prediction['strength'] * 50)),
        'reason': f'Padrão cíclico indica {prediction["dominant"]}',
        'algorithm': 'CYCLICAL'
    }

def generate_distribution_prediction():
    patterns = st.session_state.advanced_analysis.get('patterns', {})
    distribution = patterns.get('distribution', {})
    deviations = distribution.get('deviations', {})
    stats = st.session_state.current_stats
    
    if deviations.get('player', 0) > 0.2 and stats['player'] < stats['banker']:
        return {
            'type': 'PLAYER',
            'confidence': min(80, 50 + (deviations['player'] * 100)),
            'reason': 'Player abaixo da distribuição esperada',
            'algorithm': 'DISTRIBUTION'
        }
    
    if deviations.get('banker', 0) > 0.2 and stats['banker'] < stats['player']:
        return {
            'type': 'BANKER',
            'confidence': min(80, 50 + (deviations['banker'] * 100)),
            'reason': 'Banker abaixo da distribuição esperada',
            'algorithm': 'DISTRIBUTION'
        }
    
    return {
        'type': 'BALANCED',
        'confidence': 40,
        'reason': 'Distribuição próxima do esperado',
        'algorithm': 'DISTRIBUTION'
    }

def generate_pattern_prediction():
    patterns = st.session_state.advanced_analysis.get('patterns', {})
    sequences = patterns.get('sequences', {})
    top_sequences = sequences.get('topSequences', [])
    
    if not top_sequences or top_sequences[0]['probability'] <= 0.15:
        return {
            'type': 'RANDOM',
            'confidence': 25,
            'reason': 'Sem padrão sequencial forte',
            'algorithm': 'PATTERN'
        }
    
    sequence_parts = top_sequences[0]['sequence'].split('-')
    next_expected = sequence_parts[-1]
    
    return {
        'type': next_expected,
        'confidence': min(65, 35 + (top_sequences[0]['probability'] * 100)),
        'reason': f'Padrão sequencial indica {next_expected}',
        'algorithm': 'PATTERN'
    }

def calculate_overall_confidence(analysis):
    results = st.session_state.results
    if len(results) < 8:
        return min(40, len(results) * 5)

    predictions = analysis['predictions']
    if not predictions:
        return 30

    weights = {
        'REVERSION': 0.25,
        'MOMENTUM': 0.20,
        'CYCLICAL': 0.20,
        'DISTRIBUTION': 0.20,
        'PATTERN': 0.15
    }

    weighted_confidence = 0
    total_weight = 0

    for pred in predictions:
        weight = weights.get(pred['algorithm'], 0.1)
        weighted_confidence += pred['confidence'] * weight
        total_weight += weight

    base_confidence = weighted_confidence / total_weight if total_weight > 0 else 30
    data_quality_multiplier = min(1.2, 0.8 + (len(results) * 0.01))
    volatility_adjustment = 0.9 if analysis['volatility'] > 70 else 1.1
    
    confidence = min(95, max(25, base_confidence * data_quality_multiplier * volatility_adjustment))
    return round(confidence)

def determine_risk_level(analysis):
    volatility = analysis['volatility']
    confidence = analysis['confidence']

    if volatility > 80 or confidence < 40:
        return 'HIGH'
    if volatility > 60 or confidence < 55:
        return 'MEDIUM'
    return 'LOW'

def get_best_recommendation():
    results = st.session_state.results
    if len(results) < 3:
        return {
            'type': 'AGUARDAR',
            'reason': 'Coletando dados iniciais...',
            'confidence': 0,
            'color': 'gray',
            'algorithm': 'NONE'
        }

    predictions = st.session_state.advanced_analysis.get('predictions', [])
    if not predictions:
        return {
            'type': 'PLAYER',
            'reason': 'Recomendação padrão (sem comissão)',
            'confidence': 52,
            'color': 'blue',
            'algorithm': 'DEFAULT'
        }

    consensus_map = {}
    for pred in predictions:
        pred_type = pred['type']
        if pred_type not in ['WAIT', 'RANDOM', 'BALANCED']:
            consensus_map[pred_type] = consensus_map.get(pred_type, 0) + pred['confidence']

    if not consensus_map:
        return {
            'type': 'AGUARDAR',
            'reason': 'Sinais conflitantes - aguardar melhor oportunidade',
            'confidence': 35,
            'color': 'yellow',
            'algorithm': 'CONSENSUS'
        }

    best_consensus = max(consensus_map.items(), key=lambda x: x[1])
    recommended_type, total_confidence = best_consensus
    
    supporting = sum(1 for p in predictions if p['type'] == recommended_type)
    avg_confidence = min(90, total_confidence / supporting) if supporting else 0

    return {
        'type': recommended_type,
        'reason': f'Consenso de {supporting} algoritmo(s)',
        'confidence': round(avg_confidence),
        'color': 'blue' if recommended_type == 'PLAYER' else 'red' if recommended_type == 'BANKER' else 'green' if recommended_type == 'TIE' else 'yellow',
        'algorithm': 'CONSENSUS',
        'supportingAlgorithms': supporting
    }

def analyze_patterns():
    if len(st.session_state.results) < 5:
        return {}
    
    patterns = {
        'streaks': analyze_streaks(),
        'alternations': analyze_alternations(),
        'hotCold': analyze_hot_cold_numbers(),
        'distribution': analyze_distribution(),
        'correlations': analyze_correlations(),
        'sequences': analyze_sequences()
    }
    
    return patterns

def perform_advanced_analysis():
    if not st.session_state.results:
        st.session_state.current_stats = {'player':0, 'banker':0, 'tie':0, 'totalGames':0}
        return

    stats = calculate_basic_stats()
    st.session_state.current_stats = stats

    analysis = {
        'patterns': analyze_patterns(),
        'cyclicalTrends': analyze_cyclical_trends(),
        'volatility': calculate_volatility(),
        'momentum': calculate_momentum(),
        'predictions': generate_predictions(),
        'confidence': 0,
        'riskLevel': 'LOW'
    }

    analysis['confidence'] = calculate_overall_confidence(analysis)
    analysis['riskLevel'] = determine_risk_level(analysis)

    st.session_state.advanced_analysis = analysis

# Interface do usuário completa com histórico ajustado
def main():
    st.set_page_config(layout="wide", page_title="Bac Bo Analyzer PRO")
    
    # CSS personalizado ATUALIZADO
    st.markdown("""
    <style>
        .big-font {
            font-size: 50px !important;
            font-weight: bold;
            background: -webkit-linear-gradient(#eee, #333);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .recommendation-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .blue-box {
            background-color: rgba(0, 0, 255, 0.1);
            border: 1px solid blue;
        }
        .red-box {
            background-color: rgba(255, 0, 0, 0.1);
            border: 1px solid red;
        }
        .green-box {
            background-color: rgba(0, 255, 0, 0.1);
            border: 1px solid green;
        }
        .gray-box {
            background-color: rgba(128, 128, 128, 0.1);
            border: 1px solid gray;
        }
        .yellow-box {
            background-color: rgba(255, 255, 0, 0.1);
            border: 1px solid yellow;
        }
        /* HISTÓRICO ATUALIZADO - MAIS COMPACTO */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(20, 1fr);
            gap: 2px;
            margin-bottom: 15px;
        }
        .grid-item {
            aspect-ratio: 1/1;
            border-radius: 3px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-size: 8px;
            font-weight: bold;
            color: white;
            padding: 1px;
            position: relative;
        }
        .grid-item.rare::after {
            content: "★";
            position: absolute;
            top: 1px;
            right: 1px;
            color: gold;
            font-size: 6px;
        }
        .compact-table {
            font-size: 12px;
        }
        .compact-table th, .compact-table td {
            padding: 2px 5px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="big-font">🎲 BAC BO ANALYZER PRO</h1>', unsafe_allow_html=True)
    st.markdown("**Análise Inteligente com 5 Algoritmos Avançados**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🧠 IA Avançada")
    with col2:
        st.caption("⚡ Análise em Tempo Real")
    with col3:
        st.caption("🔄 Padrões Cíclicos")

    # Input Section
    st.subheader("🎲 Inserir Resultado")
    col_player, col_banker, col_button = st.columns([1,1,2])
    with col_player:
        player_score = st.number_input("Player (2-12)", min_value=2, max_value=12, step=1, key="player")
    with col_banker:
        banker_score = st.number_input("Banker (2-12)", min_value=2, max_value=12, step=1, key="banker")
    with col_button:
        st.write("")
        st.write("")
        if st.button("Adicionar", type="primary"):
            add_result(player_score, banker_score)
    
    st.caption(f"Total de jogos analisados: {len(st.session_state.results)}")

    # Layout principal
    col_left, col_right = st.columns([1,2])

    with col_left:
        # Recomendação
        st.subheader("🎯 Recomendação IA")
        recommendation = get_best_recommendation()
        
        # Caixa de recomendação
        box_class = ""
        if recommendation['color'] == 'blue':
            box_class = "blue-box"
        elif recommendation['color'] == 'red':
            box_class = "red-box"
        elif recommendation['color'] == 'green':
            box_class = "green-box"
        elif recommendation['color'] == 'gray':
            box_class = "gray-box"
        else:
            box_class = "yellow-box"

        st.markdown(f'<div class="recommendation-box {box_class}">', unsafe_allow_html=True)
        st.markdown(f"**{recommendation['type']}**")
        st.write(recommendation['reason'])
        st.write(f"**Confiança:** {recommendation['confidence']}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Nível de risco
        risk_level = st.session_state.advanced_analysis['riskLevel']
        risk_color = "🟢" if risk_level == 'LOW' else "🟡" if risk_level == 'MEDIUM' else "🔴"
        st.metric("📉 Risco", f"{risk_color} {risk_level}")
        st.metric("📊 Volatilidade", f"{st.session_state.advanced_analysis['volatility']}%")

        # Estatísticas Avançadas
        st.subheader("📈 Estatísticas Avançadas")
        stats = st.session_state.current_stats
        st.metric("🔵 Player", stats['player'])
        st.metric("🔴 Banker", stats['banker'])
        st.metric("🟢 Tie", stats['tie'])

        # Momentum
        momentum = st.session_state.advanced_analysis.get('momentum', {})
        if momentum and isinstance(momentum, dict):
            direction_icon = "⬆️" if momentum.get('direction') == 'PLAYER' else "⬇️"
            st.metric("🚀 Momentum", 
                     f"{direction_icon} {momentum.get('direction', 'N/A')}",
                     f"Força: {round(momentum.get('strength', 0)*100)}%")

        # Sequência atual
        patterns = st.session_state.advanced_analysis.get('patterns', {})
        streaks = patterns.get('streaks', {})
        current_streak = streaks.get('currentStreak', {})
        if current_streak.get('count', 0) > 1:
            streak_type = "🔵 Player" if current_streak['type'] == 'player' else \
                         "🔴 Banker" if current_streak['type'] == 'banker' else "🟢 Tie"
            st.metric("🔥 Sequência Atual", 
                     f"{current_streak['count']} jogos", 
                     streak_type)

    with col_right:
        # Análise Multi-Algoritmo
        st.subheader("🤖 Análise Multi-Algoritmo")
        predictions = st.session_state.advanced_analysis.get('predictions', [])
        
        if predictions:
            cols = st.columns(2)
            for i, pred in enumerate(predictions[:4]):
                with cols[i % 2]:
                    with st.container(border=True):
                        st.caption(pred['algorithm'])
                        st.subheader(pred['type'])
                        st.progress(pred['confidence']/100, text=f"{pred['confidence']}%")
                        st.caption(pred['reason'])
        else:
            st.info("Coletando dados para análise...")

        # Histórico de Resultados ATUALIZADO
        st.subheader("🕒 Histórico de Resultados")
        if st.session_state.results:
            # Roadmap compacto
            st.markdown('<div class="grid-container">', unsafe_allow_html=True)
            for i, result in enumerate(st.session_state.results[:80]):  # Mostrar mais resultados
                rare_class = " rare" if result['surprise'] > 80 else ""
                st.markdown(
                    f'<div class="grid-item{rare_class}" style="background-color: {result["color"]}" title="Jogo: {result["gameNumber"]} | Player: {result["player"]} | Banker: {result["banker"]} | Surpresa: {result["surprise"]}%">'
                    f'{result["player"]}|{result["banker"]}'
                    '</div>', 
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # Tabela detalhada compacta
            with st.expander("📋 Ver últimos 15 resultados detalhados", expanded=False):
                recent = st.session_state.results[:15]
                table_data = []
                for result in recent:
                    table_data.append({
                        '#': result['gameNumber'],
                        'Hora': result['timestamp'],
                        'Player': result['player'],
                        'Banker': result['banker'],
                        'Resultado': result['outcome'],
                        'Surpresa': f"{result['surprise']}%"
                    })
                
                # Usando uma tabela mais compacta
                st.dataframe(
                    pd.DataFrame(table_data),
                    height=300,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "#": st.column_config.NumberColumn(width="small"),
                        "Hora": st.column_config.TextColumn(width="small"),
                        "Player": st.column_config.NumberColumn(width="small"),
                        "Banker": st.column_config.NumberColumn(width="small"),
                        "Resultado": st.column_config.TextColumn(width="small"),
                        "Surpresa": st.column_config.TextColumn(width="small")
                    }
                )
        else:
            st.info("Nenhum resultado inserido ainda")

    # Seções avançadas
    if st.session_state.results and len(st.session_state.results) > 10:
        st.divider()
        st.subheader("🔍 Análise Avançada")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("🎰 Padrões", "Detecção")
            streaks = st.session_state.advanced_analysis['patterns'].get('streaks', {})
            if streaks:
                st.write(f"🔵 Max Player: {streaks['maxStreaks']['player']}")
                st.write(f"🔴 Max Banker: {streaks['maxStreaks']['banker']}")
                st.write(f"🔄 Alternância: {round(st.session_state.advanced_analysis['patterns']['alternations'].get('rate', 0)*100)}%")
        
        with cols[1]:
            st.metric("🔥 Números Quentes", "Frequência")
            hot_cold = st.session_state.advanced_analysis['patterns'].get('hotCold', {})
            if hot_cold:
                st.write("🔵 Player:")
                if hot_cold['player'].get('hot'):
                    st.write(", ".join(str(i['num']) for i in hot_cold['player']['hot']))
                st.write("🔴 Banker:")
                if hot_cold['banker'].get('hot'):
                    st.write(", ".join(str(i['num']) for i in hot_cold['banker']['hot']))
        
        with cols[2]:
            st.metric("🔄 Ciclos", "Padrões")
            cycle5 = st.session_state.advanced_analysis['cyclicalTrends'].get('cycle5', {})
            if cycle5:
                st.write(f"Fase: {cycle5.get('currentPhase', 0)+1}/5")
                pred = cycle5.get('predictedNext', {})
                if pred:
                    st.write(f"Próximo: {pred.get('dominant', 'N/A')}")
                    st.write(f"Conf: {round(pred.get('strength', 0) * 100)}%")
        
        with cols[3]:
            st.m metric("🤖 Performance IA", "Sistema")
            st.write(f"Confiança: {round(st.session_state.advanced_analysis['confidence'])}%")
            st.write(f"Algoritmos: {len(st.session_state.advanced_analysis['predictions'])}")
            st.write(f"Dados: {len(st.session_state.results)}")
            status = "🟢 ALTA" if st.session_state.advanced_analysis['confidence'] >= 70 else \
                     "🟡 MÉDIA" if st.session_state.advanced_analysis['confidence'] >= 50 else "🔴 BAIXA"
            st.write(status)

    # Footer
    st.divider()
    st.caption("Sistema de análise avançada com 5 algoritmos de IA para Bac Bo Live")
    st.caption("Baseado em análise estatística, padrões cíclicos e probabilidades matemáticas")

if __name__ == "__main__":
    main()
