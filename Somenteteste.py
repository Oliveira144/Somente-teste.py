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
    except ValueError:
        st.error("Por favor, insira números válidos.")
        return

    if pScore < 2 or pScore > 12 or bScore < 2 or bScore > 12:
        st.error("As pontuações devem estar entre 2 e 12.")
        return

    if pScore > bScore:
        outcome = 'PLAYER'
        color = '#4285F4' # Blue for Player
    elif bScore > pScore:
        outcome = 'BANKER'
        color = '#EA4335' # Red for Banker
    else:
        outcome = 'TIE'
        color = '#34A853' # Green for Tie

    new_result = {
        'id': time.time(),
        'player': pScore,
        'banker': bScore,
        'outcome': outcome,
        'color': color,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'playerProb': DICE_PROBABILITIES.get(pScore, 0),
        'bankerProb': DICE_PROBABILITIES.get(bScore, 0),
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
            if current_streak['type'] and current_streak['count'] > 0:
                streaks[current_streak['type']].append(current_streak['count'])
            current_streak = {'type': outcome, 'count': 1}

    if current_streak['type'] and current_streak['count'] > 0:
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
    if len(results) < 2:
        return {'rate': 0, 'pattern': 'NONE', 'alternations': 0, 'consecutives': 0}

    alternations = 0
    
    for i in range(len(results) - 1, 0, -1):
        if results[i]['outcome'] != results[i-1]['outcome']:
            alternations += 1

    consecutives = (len(results) - 1) - alternations
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
        if not entries:
            return {'hot': [], 'cold': []}
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
        return {'deviations': {}, 'expected': {'player': 0, 'banker': 0, 'tie': 0}}
    
    expected_player_ratio = 0.486
    expected_banker_ratio = 0.486
    expected_tie_ratio = 0.028

    expected_player = n * expected_player_ratio
    expected_banker = n * expected_banker_ratio
    expected_tie = n * expected_tie_ratio

    stats = st.session_state.current_stats
    deviations = {
        'player': abs(stats['player'] - expected_player) / (expected_player or 1),
        'banker': abs(stats['banker'] - expected_banker) / (expected_banker or 1),
        'tie': abs(stats['tie'] - expected_tie) / (expected_tie or 1)
    }

    return {'deviations': deviations, 'expected': {'player': expected_player, 'banker': expected_banker, 'tie': expected_tie}}

def analyze_correlations():
    results = st.session_state.results
    if len(results) < 2:
        return {'playerNumberCorrelation': 0, 'bankerNumberCorrelation': 0, 'outcomeCorrelation': 0}
    
    player_corr_sum = 0
    banker_corr_sum = 0
    
    n_correlate = min(len(results) - 1, 20)

    for i in range(1, n_correlate + 1):
        curr_idx = i - 1
        prev_idx = i
        
        if curr_idx >= 0 and prev_idx < len(results):
            curr = results[curr_idx]
            prev = results[prev_idx]
            player_corr_sum += curr['player'] * prev['player']
            banker_corr_sum += curr['banker'] * prev['banker']

    player_corr = player_corr_sum / n_correlate if n_correlate > 0 else 0
    banker_corr = banker_corr_sum / n_correlate if n_correlate > 0 else 0

    return {
        'playerNumberCorrelation': player_corr,
        'bankerNumberCorrelation': banker_corr,
        'outcomeCorrelation': analyze_outcome_correlation()
    }

def analyze_outcome_correlation():
    results = st.session_state.results
    if len(results) < 4:
        return 0

    matches = 0
    n_lag_check = min(len(results) - 3, 10) 

    for i in range(3, 3 + n_lag_check):
        if results[i]['outcome'] == results[i-3]['outcome']:
            matches += 1

    return matches / n_lag_check if n_lag_check > 0 else 0

def analyze_sequences():
    results = st.session_state.results
    if len(results) < 2:
        return {'sequences': {}, 'topSequences': []}

    sequences = defaultdict(int)
    
    chronological_results = results[::-1] 

    for length in range(2, min(5, len(chronological_results) + 1)):
        for i in range(len(chronological_results) - length + 1):
            seq = '-'.join(r['outcome'] for r in chronological_results[i:i+length])
            sequences[seq] += 1

    top_sequences = sorted([(seq, count) for seq, count in sequences.items()], 
                          key=lambda x: x[1], reverse=True)[:5]
    
    top_sequences_formatted = [{
        'sequence': seq,
        'count': count,
        'probability': count / len(chronological_results) if len(chronological_results) > 0 else 0
    } for seq, count in top_sequences]

    return {'sequences': sequences, 'topSequences': top_sequences_formatted}

def analyze_cyclical_trends():
    results = st.session_state.results
    if len(results) < 12:
        return {}

    cycles = [3, 5, 7, 10]
    cyclical_data = {}

    chronological_results = results[::-1] 

    for cycle in cycles:
        buckets = [{'player': 0, 'banker': 0, 'tie': 0} for _ in range(cycle)]
        
        for i, result in enumerate(chronological_results):
            bucket_idx = i % cycle
            outcome = result['outcome'].lower()
            buckets[bucket_idx][outcome] += 1

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

        current_phase_idx = len(chronological_results) % cycle
        predicted_next_phase = dominant_phases[current_phase_idx] if dominant_phases else None

        cyclical_data[f'cycle{cycle}'] = {
            'buckets': buckets,
            'dominantPhases': dominant_phases,
            'currentPhase': current_phase_idx,
            'predictedNext': predicted_next_phase
        }

    return cyclical_data

def calculate_volatility():
    results = st.session_state.results
    if len(results) < 2:
        return 0

    recent = results[:20]
    changes = 0
    total_surprise = 0

    for i in range(1, len(recent)):
        if recent[i]['outcome'] != recent[i-1]['outcome']:
            changes += 1
        total_surprise += recent[i]['surprise']

    change_rate = changes / (len(recent) - 1) if len(recent) > 1 else 0
    avg_surprise = total_surprise / len(recent) if recent else 0
    volatility = min(100, (change_rate * 50) + (avg_surprise * 0.5))
    
    return round(volatility)

def calculate_momentum():
    results = st.session_state.results
    if len(results) < 8:
        return {
            'direction': 'N/A',
            'strength': 0,
            'playerMomentum': 0,
            'bankerMomentum': 0
        }

    recent = results[:8]
    weights = [0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005] 
    
    player_momentum = 0
    banker_momentum = 0

    for i, result in enumerate(recent):
        if i < len(weights):
            weight = weights[i]
            if result['outcome'] == 'PLAYER':
                player_momentum += weight
            elif result['outcome'] == 'BANKER':
                banker_momentum += weight

    strength = abs(player_momentum - banker_momentum)
    direction = 'PLAYER' if player_momentum > banker_momentum else 'BANKER'
    if player_momentum == banker_momentum:
        direction = 'TIE/BALANCED'

    return {
        'direction': direction,
        'strength': strength,
        'playerMomentum': player_momentum,
        'bankerMomentum': banker_momentum
    }

def generate_predictions():
    predictions = []
    
    predictions.append(generate_reversion_prediction())
    predictions.append(generate_momentum_prediction())
    predictions.append(generate_cyclical_prediction())
    predictions.append(generate_distribution_prediction())
    predictions.append(generate_pattern_prediction())
    
    return [p for p in predictions if p['confidence'] > 30]

def generate_reversion_prediction():
    results = st.session_state.results
    if len(results) < 5:
        return {'type': 'WAIT', 'confidence': 0, 'reason': 'Dados insuficientes', 'algorithm': 'REVERSION'}

    recent = results[:5]
    
    if not recent:
        return {'type': 'WAIT', 'confidence': 0, 'reason': 'Dados insuficientes', 'algorithm': 'REVERSION'}

    last_outcome = recent[0]['outcome']
    
    consecutive_count = 0
    for r in recent:
        if r['outcome'] == last_outcome:
            consecutive_count += 1
        else:
            break

    if consecutive_count >= 4:
        opposite = 'BANKER' if last_outcome == 'PLAYER' else 'PLAYER' if last_outcome == 'BANKER' else 'TIE'
        confidence = min(85, 45 + (consecutive_count * 10)) 
        return {
            'type': opposite,
            'confidence': confidence,
            'reason': f'Forte tendência de reversão após {consecutive_count} {last_outcome}s consecutivos.',
            'algorithm': 'REVERSION'
        }

    if consecutive_count >= 3:
        opposite = 'BANKER' if last_outcome == 'PLAYER' else 'PLAYER' if last_outcome == 'BANKER' else 'TIE'
        return {
            'type': opposite,
            'confidence': 60,
            'reason': f'Provável reversão após {consecutive_count} {last_outcome}s.',
            'algorithm': 'REVERSION'
        }

    return {'type': 'WAIT', 'confidence': 25, 'reason': 'Sem padrão de reversão claro', 'algorithm': 'REVERSION'}

def generate_momentum_prediction():
    momentum = st.session_state.advanced_analysis.get('momentum', {})
    if not momentum or momentum.get('direction') == 'N/A' or momentum.get('strength', 0) == 0:
        return {'type': 'WAIT', 'confidence': 0, 'reason': 'Momentum indisponível ou fraco', 'algorithm': 'MOMENTUM'}
    
    if momentum.get('strength', 0) > 0.4:
        return {
            'type': momentum['direction'],
            'confidence': min(75, 40 + int(momentum['strength'] * 100)),
            'reason': f'Momento forte para {momentum["direction"]}',
            'algorithm': 'MOMENTUM'
        }
    elif momentum.get('strength', 0) > 0.2:
        return {
            'type': momentum['direction'],
            'confidence': min(60, 30 + int(momentum['strength'] * 100)),
            'reason': f'Momento moderado para {momentum["direction"]}',
            'algorithm': 'MOMENTUM'
        }

    return {
        'type': 'BALANCED',
        'confidence': 35,
        'reason': 'Momento equilibrado',
        'algorithm': 'MOMENTUM'
    }


def generate_cyclical_prediction():
    cyclical_trends = st.session_state.advanced_analysis.get('cyclicalTrends', {})
    cycle5_data = cyclical_trends.get('cycle5', {})
    predicted_next_phase = cycle5_data.get('predictedNext', {})
    
    if not predicted_next_phase or predicted_next_phase.get('strength', 0) <= 0.5:
        return {
            'type': 'RANDOM',
            'confidence': 30,
            'reason': 'Sem padrão cíclico claro ou fraco',
            'algorithm': 'CYCLICAL'
        }
    
    return {
        'type': predicted_next_phase['dominant'],
        'confidence': min(70, 30 + int(predicted_next_phase['strength'] * 50)),
        'reason': f'Padrão cíclico (ciclo 5) indica {predicted_next_phase["dominant"]}',
        'algorithm': 'CYCLICAL'
    }

def generate_distribution_prediction():
    patterns = st.session_state.advanced_analysis.get('patterns', {})
    distribution = patterns.get('distribution', {})
    deviations = distribution.get('deviations', {})
    stats = st.session_state.current_stats
    
    if stats['totalGames'] > 10 and deviations.get('player', 0) > 0.2 and \
       (stats['player'] / stats['totalGames'] < (distribution['expected']['player'] / stats['totalGames'] if stats['totalGames'] > 0 else 0) * 0.8):
        return {
            'type': 'PLAYER',
            'confidence': min(80, 50 + int(deviations['player'] * 100)),
            'reason': 'Player significativamente abaixo da distribuição esperada',
            'algorithm': 'DISTRIBUTION'
        }
    
    if stats['totalGames'] > 10 and deviations.get('banker', 0) > 0.2 and \
       (stats['banker'] / stats['totalGames'] < (distribution['expected']['banker'] / stats['totalGames'] if stats['totalGames'] > 0 else 0) * 0.8):
        return {
            'type': 'BANKER',
            'confidence': min(80, 50 + int(deviations['banker'] * 100)),
            'reason': 'Banker significativamente abaixo da distribuição esperada',
            'algorithm': 'DISTRIBUTION'
        }
    
    return {
        'type': 'BALANCED',
        'confidence': 40,
        'reason': 'Distribuição próxima do esperado ou dados insuficientes para desvio significativo',
        'algorithm': 'DISTRIBUTION'
    }

def generate_pattern_prediction():
    patterns = st.session_state.advanced_analysis.get('patterns', {})
    sequences_analysis = patterns.get('sequences', {})
    top_sequences = sequences_analysis.get('topSequences', [])
    
    if not top_sequences or top_sequences[0]['probability'] <= 0.20:
        return {
            'type': 'RANDOM',
            'confidence': 25,
            'reason': 'Sem padrão sequencial forte o suficiente',
            'algorithm': 'PATTERN'
        }
    
    most_frequent_seq = top_sequences[0]['sequence']
    sequence_parts = most_frequent_seq.split('-')
    next_expected = sequence_parts[-1]

    return {
        'type': next_expected,
        'confidence': min(65, 35 + int(top_sequences[0]['probability'] * 100)),
        'reason': f'Padrão sequencial comum: "{most_frequent_seq}"',
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
        weight = weights.get(pred['algorithm'], 0)
        weighted_confidence += pred['confidence'] * weight
        total_weight += weight

    base_confidence = weighted_confidence / total_weight if total_weight > 0 else 30
    
    data_quality_multiplier = min(1.2, 0.8 + (len(results) * 0.01))
    volatility_adjustment = 0.9 if analysis['volatility'] > 70 else 1.1 if analysis['volatility'] < 30 else 1

    confidence = min(95, max(25, base_confidence * data_quality_multiplier * volatility_adjustment))
    return round(confidence)

def determine_risk_level(analysis):
    volatility = analysis['volatility']
    confidence = analysis['confidence']

    if volatility > 75 or confidence < 45:
        return 'HIGH'
    if volatility > 55 or confidence < 60:
        return 'MEDIUM'
    return 'LOW'

def get_best_recommendation():
    results = st.session_state.results
    if len(results) < 3:
        return {
            'type': 'AGUARDAR',
            'reason': 'Coletando dados iniciais para análise...',
            'confidence': 0,
            'color': 'gray',
            'algorithm': 'NONE'
        }

    predictions = st.session_state.advanced_analysis.get('predictions', [])
    
    if not predictions:
        return {
            'type': 'PLAYER',
            'reason': 'Nenhuma previsão forte no momento, recomendação padrão (sem comissão)',
            'confidence': 50,
            'color': '#4285F4',
            'algorithm': 'DEFAULT'
        }

    consensus_map = {}
    for pred in predictions:
        pred_type = pred['type']
        if pred_type not in ['WAIT', 'RANDOM', 'BALANCED'] and pred['confidence'] > 50:
            consensus_map[pred_type] = consensus_map.get(pred_type, 0) + pred['confidence']

    if not consensus_map:
        return {
            'type': 'AGUARDAR',
            'reason': 'Sinais conflitantes ou fracos - aguardar melhor oportunidade',
            'confidence': 35,
            'color': 'yellow',
            'algorithm': 'CONSENSUS'
        }

    best_consensus_type, total_consensus_confidence = max(consensus_map.items(), key=lambda x: x[1])
    
    supporting_algos = sum(1 for p in predictions if p['type'] == best_consensus_type and p['confidence'] > 50)
    avg_confidence = min(95, total_consensus_confidence / supporting_algos) if supporting_algos else 0

    return {
        'type': best_consensus_type,
        'reason': f'Consenso de {supporting_algos} algoritmo(s) para {best_consensus_type}',
        'confidence': round(avg_confidence),
        'color': '#4285F4' if best_consensus_type == 'PLAYER' else '#EA4335' if best_consensus_type == 'BANKER' else '#34A853' if best_consensus_type == 'TIE' else 'yellow',
        'algorithm': 'CONSENSUS',
        'supportingAlgorithms': supporting_algos
    }

def analyze_patterns():
    if len(st.session_state.results) < 2:
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
        st.session_state.advanced_analysis = {
            'patterns': {},
            'predictions': [],
            'confidence': 0,
            'volatility': 0,
            'momentum': {'direction': 'N/A', 'strength': 0, 'playerMomentum': 0, 'bankerMomentum': 0},
            'cyclicalTrends': {},
            'riskLevel': 'LOW'
        }
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

# Interface do usuário com histórico compacto
def main():
    st.set_page_config(layout="wide", page_title="Bac Bo Analyzer PRO")
    
    # CSS personalizado para o layout em linha de 6 resultados
    st.markdown("""
    <style>
        .recommendation-box {
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 15px;
            background-color: #1e2130;
            border: 2px solid #4a4e69;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .history-grid {
            display: flex;
            flex-wrap: wrap; /* Permite que os itens quebrem para a próxima linha */
            gap: 8px; /* Espaço entre os itens */
            padding: 10px 0;
            justify-content: flex-start; /* Alinha os itens ao início */
            width: 100%; /* Garante que o container use a largura total para a quebra */
        }
        .history-item {
            /* 100% dividido por 6 itens, subtraindo o gap, para ter 6 por linha */
            width: calc((100% / 6) - 8px); 
            /* Se a tela for muito estreita e calc() não for suficiente, 
               pode-se adicionar um min-width para evitar que fiquem minúsculos 
               ou um max-width para telas muito largas. */
            min-width: 50px; /* Garante um tamanho mínimo para visualização em telas pequenas */
            max-width: 60px; /* Limite máximo para o item, ajuste conforme necessário */
            height: 60px; /* Altura do item */
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-size: 1.1em;
            font-weight: bold;
            color: white;
            padding: 2px;
            box-sizing: border-box;
            flex-shrink: 0; /* Evita que os itens encolham abaixo do seu tamanho calculado */
        }
        .score-display {
            font-size: 0.9em;
            line-height: 1.2;
        }
        .player-score {
            color: white;
        }
        .banker-score {
            color: white;
        }
        .header-section {
            background-color: #1e2130;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        .header-section h3 {
            margin: 0;
            color: #fff;
            margin-left: 10px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 50px;
            font-size: 1.1em;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            height: 50px;
            font-size: 1.1em;
            text-align: center;
        }
        .stMetric {
            background-color: #1e2130;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            color: #fff;
            border: 1px solid #4a4e69;
        }
        .stMetric > div > div > div > div > div {
            color: #fff !important;
        }
        .stMetric > div > div > div {
            color: #fff !important;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #fff;
        }
        p {
            color: #ccc;
        }
        .stAlert {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
        <div class="header-section">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
            <h3>Historico de Resultados</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # History Display
    st.markdown('<div class="history-grid">', unsafe_allow_html=True)
    # Exibe todos os resultados (até 100)
    for result in st.session_state.results: 
        st.markdown(f"""
        <div class="history-item" style="background-color: {result['color']};">
            <div class="score-display player-score">{result['player']}</div>
            <div class="score-display banker-score">{result['banker']}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Input for new results
    st.header("Adicionar Novo Resultado")
    col1, col2 = st.columns(2)
    with col1:
        player_input = st.text_input("Pontuação do Player", key="player_score_input")
    with col2:
        banker_input = st.text_input("Pontuação do Banker", key="banker_score_input")

    if st.button("Adicionar Resultado"):
        if player_input and banker_input:
            add_result(player_input, banker_input)
            st.rerun()
        else:
            st.error("Por favor, preencha ambas as pontuações.")

    st.markdown("---")

    # Current Statistics
    st.header("Estatísticas Atuais")
    stats = st.session_state.current_stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total de Jogos", value=stats['totalGames'])
    with col2:
        st.metric(label="Player", value=f"{stats['player']} ({stats['player']/stats['totalGames']*100:.1f}%)" if stats['totalGames'] > 0 else "0 (0%)")
    with col3:
        st.metric(label="Banker", value=f"{stats['banker']} ({stats['banker']/stats['totalGames']*100:.1f}%)" if stats['totalGames'] > 0 else "0 (0%)")
    with col4:
        st.metric(label="Tie", value=f"{stats['tie']} ({stats['tie']/stats['totalGames']*100:.1f}%)" if stats['totalGames'] > 0 else "0 (0%)")

    st.markdown("---")

    # Recommendation
    st.header("Recomendação Inteligente")
    recommendation = get_best_recommendation()
    st.markdown(f"""
    <div class="recommendation-box" style="background-color: {recommendation['color']};">
        <h4>Ação Recomendada: {recommendation['type']}</h4>
        <p>{recommendation['reason']}</p>
        <p>Confiança: {recommendation['confidence']}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Advanced Analysis Details (optional, could be in an expander)
    st.markdown("---")
    st.header("Análise Avançada")
    analysis = st.session_state.advanced_analysis

    if analysis and st.session_state.results:
        st.subheader("Padrões e Tendências")
        st.json(analysis['patterns'])

        st.subheader("Tendências Cíclicas")
        st.json(analysis['cyclicalTrends'])

        st.subheader("Volatilidade e Momento")
        col_v, col_m = st.columns(2)
        with col_v:
            st.metric("Volatilidade do Jogo", f"{analysis['volatility']}%")
        with col_m:
            momentum_dir = analysis['momentum'].get('direction', 'N/A')
            momentum_str = analysis['momentum'].get('strength', 0)
            st.metric("Momento Atual", f"{momentum_dir} (Força: {momentum_str:.2f})")
        
        st.subheader("Nível de Risco")
        risk_color = "green" if analysis['riskLevel'] == 'LOW' else "orange" if analysis['riskLevel'] == 'MEDIUM' else "red"
        st.markdown(f"<p style='color:{risk_color}; font-weight:bold;'>Risco: {analysis['riskLevel']}</p>", unsafe_allow_html=True)

        st.subheader("Previsões por Algoritmo")
        for pred in analysis['predictions']:
            st.write(f"**Algoritmo:** {pred['algorithm']} - **Previsão:** {pred['type']} - **Confiança:** {pred['confidence']}% - **Razão:** {pred['reason']}")
    elif not st.session_state.results:
        st.info("Adicione resultados para ver a análise avançada.")

if __name__ == '__main__':
    main()
