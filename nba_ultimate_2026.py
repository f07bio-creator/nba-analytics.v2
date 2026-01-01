import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import json
import requests
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DO SISTEMA
# ============================================================================

st.set_page_config(
    page_title="NBA ULTIMATE ANALYTICS 2025-26",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS profissional
st.markdown("""
<style>
    /* Paleta NBA */
    :root {
        --nba-blue: #17408B;
        --nba-red: #C9082A;
        --nba-white: #FFFFFF;
        --nba-gray: #F8F9FA;
        --nba-dark: #212529;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, var(--nba-blue), var(--nba-red));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .team-card {
        background: linear-gradient(135deg, var(--nba-blue) 0%, var(--nba-red) 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(23, 64, 139, 0.3);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--nba-blue), var(--nba-red));
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, var(--nba-blue), var(--nba-red));
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 5px;
    }
    
    .high-confidence { background: linear-gradient(90deg, #28a745, #20c997); color: white; }
    .medium-confidence { background: linear-gradient(90deg, #ffc107, #fd7e14); color: white; }
    .low-confidence { background: linear-gradient(90deg, #dc3545, #e83e8c); color: white; }
    
    /* Animações */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Estilo para abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--nba-blue), var(--nba-red));
        color: white !important;
    }
    
    /* Estilo para alertas de validação */
    .validation-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. SISTEMA DE DADOS NBA 2025-26
# ============================================================================

class NBA2026DataEngine:
    """Motor de dados para temporada 2025-26"""
    
    def __init__(self):
        self.season = "2025-26"
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nba.com/',
            'Origin': 'https://www.nba.com',
            'Connection': 'keep-alive',
        }
        self.initialize_database()
        self.cached_data = {}
        
    def initialize_database(self):
        """Inicializa banco de dados SQLite"""
        self.conn = sqlite3.connect('nba_2026_analytics.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Tabela de estatísticas dos times
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats_2026 (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT,
                data_json TEXT,
                last_updated TIMESTAMP,
                season TEXT
            )
        ''')
        
        # Tabela de jogos
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_logs_2026 (
                game_id TEXT PRIMARY KEY,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                date DATE,
                data_json TEXT
            )
        ''')
        
        # Tabela de análises salvas
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_analyses (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                matchup TEXT,
                prediction_data TEXT,
                accuracy_score REAL,
                features_used TEXT
            )
        ''')
        
        self.conn.commit()
    
    def get_all_teams(self) -> List[str]:
        """Retorna todos os times da NBA 2025-26"""
        teams_2025_26 = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
            "Utah Jazz", "Washington Wizards"
        ]
        return teams_2025_26
    
    def get_team_stats(self, team_name: str, use_cache: bool = True) -> Dict:
        """Obtém estatísticas do time para 2025-26"""
        
        # Verificar cache primeiro
        if use_cache:
            cache_key = f"{team_name}_{self.season}"
            if cache_key in self.cached_data:
                cached_time = self.cached_data[cache_key]['timestamp']
                if datetime.now() - cached_time < timedelta(hours=6):
                    return self.cached_data[cache_key]['data']
            
            # Verificar banco de dados
            self.cursor.execute(
                "SELECT data_json, last_updated FROM team_stats_2026 WHERE team_name = ? AND season = ?",
                (team_name, self.season)
            )
            result = self.cursor.fetchone()
            
            if result:
                data_json, last_updated = result
                if datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S') > datetime.now() - timedelta(hours=24):
                    return json.loads(data_json)
        
        try:
            # Simular dados da API 2025-26 (atualizar quando a temporada começar)
            stats = self._simulate_2026_stats(team_name)
            
            # Salvar no cache
            cache_key = f"{team_name}_{self.season}"
            self.cached_data[cache_key] = {
                'data': stats,
                'timestamp': datetime.now()
            }
            
            # Salvar no banco de dados
            self.cursor.execute('''
                INSERT OR REPLACE INTO team_stats_2026 
                (team_name, data_json, last_updated, season)
                VALUES (?, ?, ?, ?)
            ''', (team_name, json.dumps(stats), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.season))
            
            self.conn.commit()
            
            return stats
            
        except Exception as e:
            st.warning(f"⚠️ Dados simulados para {team_name} (API 2025-26 disponível em Outubro)")
            return self._get_backup_stats(team_name)
    
    def _simulate_2026_stats(self, team_name: str) -> Dict:
        """Simula estatísticas para 2025-26 baseado em projeções"""
        
        # Projeções para 2025-26 baseadas em tendências
        projections_2026 = {
            "Boston Celtics": {"win_pct": 0.72, "off_rating": 121.5, "def_rating": 110.8, "pace": 98.5},
            "Oklahoma City Thunder": {"win_pct": 0.68, "off_rating": 120.8, "def_rating": 112.5, "pace": 101.0},
            "Minnesota Timberwolves": {"win_pct": 0.66, "off_rating": 116.2, "def_rating": 108.9, "pace": 97.8},
            "Denver Nuggets": {"win_pct": 0.65, "off_rating": 118.8, "def_rating": 113.2, "pace": 98.9},
            "Milwaukee Bucks": {"win_pct": 0.64, "off_rating": 120.9, "def_rating": 115.8, "pace": 102.0},
            "New York Knicks": {"win_pct": 0.62, "off_rating": 117.4, "def_rating": 112.8, "pace": 95.8},
            "Philadelphia 76ers": {"win_pct": 0.60, "off_rating": 118.2, "def_rating": 114.5, "pace": 99.5},
            "LA Clippers": {"win_pct": 0.59, "off_rating": 119.1, "def_rating": 115.3, "pace": 97.3},
            "Dallas Mavericks": {"win_pct": 0.58, "off_rating": 118.5, "def_rating": 115.7, "pace": 100.2},
            "Indiana Pacers": {"win_pct": 0.57, "off_rating": 122.1, "def_rating": 118.3, "pace": 103.1},
            "Los Angeles Lakers": {"win_pct": 0.56, "off_rating": 116.8, "def_rating": 114.2, "pace": 101.0},
            "Golden State Warriors": {"win_pct": 0.55, "off_rating": 117.2, "def_rating": 114.8, "pace": 100.5},
            "Miami Heat": {"win_pct": 0.54, "off_rating": 114.2, "def_rating": 112.5, "pace": 96.3},
            "Phoenix Suns": {"win_pct": 0.53, "off_rating": 116.5, "def_rating": 115.1, "pace": 98.7},
            "Cleveland Cavaliers": {"win_pct": 0.52, "off_rating": 115.3, "def_rating": 113.2, "pace": 96.5},
            "Orlando Magic": {"win_pct": 0.51, "off_rating": 114.8, "def_rating": 112.1, "pace": 97.9},
            "New Orleans Pelicans": {"win_pct": 0.50, "off_rating": 115.6, "def_rating": 113.9, "pace": 99.0},
            "Sacramento Kings": {"win_pct": 0.49, "off_rating": 117.1, "def_rating": 116.0, "pace": 101.2},
            "Atlanta Hawks": {"win_pct": 0.48, "off_rating": 118.5, "def_rating": 119.2, "pace": 103.3},
            "Houston Rockets": {"win_pct": 0.47, "off_rating": 114.5, "def_rating": 113.1, "pace": 98.2},
            "Chicago Bulls": {"win_pct": 0.46, "off_rating": 112.4, "def_rating": 114.0, "pace": 98.0},
            "Toronto Raptors": {"win_pct": 0.45, "off_rating": 113.8, "def_rating": 116.9, "pace": 101.3},
            "Brooklyn Nets": {"win_pct": 0.44, "off_rating": 113.1, "def_rating": 114.8, "pace": 99.1},
            "Utah Jazz": {"win_pct": 0.43, "off_rating": 116.3, "def_rating": 119.5, "pace": 101.0},
            "Memphis Grizzlies": {"win_pct": 0.42, "off_rating": 106.5, "def_rating": 113.2, "pace": 97.1},
            "Charlotte Hornets": {"win_pct": 0.41, "off_rating": 109.8, "def_rating": 118.3, "pace": 99.0},
            "Portland Trail Blazers": {"win_pct": 0.40, "off_rating": 108.2, "def_rating": 116.7, "pace": 98.5},
            "San Antonio Spurs": {"win_pct": 0.39, "off_rating": 111.8, "def_rating": 117.5, "pace": 102.5},
            "Washington Wizards": {"win_pct": 0.38, "off_rating": 113.2, "def_rating": 121.8, "pace": 103.8},
            "Detroit Pistons": {"win_pct": 0.37, "off_rating": 109.8, "def_rating": 118.5, "pace": 100.0}
        }
        
        if team_name in projections_2026:
            proj = projections_2026[team_name]
        else:
            proj = {"win_pct": 0.50, "off_rating": 115.0, "def_rating": 115.0, "pace": 100.0}
        
        # Gerar estatísticas detalhadas baseadas nas projeções
        stats = {
            'name': team_name,
            'season': self.season,
            'win_pct': proj['win_pct'],
            'off_rating': proj['off_rating'],
            'def_rating': proj['def_rating'],
            'net_rating': proj['off_rating'] - proj['def_rating'],
            'pace': proj['pace'],
            'points': proj['off_rating'] / 100 * proj['pace'],
            'opp_points': proj['def_rating'] / 100 * proj['pace'],
            
            # Estatísticas avançadas
            'effective_fg_pct': 0.545 + (proj['off_rating'] - 115) * 0.003,
            'true_shooting_pct': 0.580 + (proj['off_rating'] - 115) * 0.0025,
            'off_reb_pct': 0.25 + (proj['win_pct'] - 0.5) * 0.1,
            'def_reb_pct': 0.75 - (proj['win_pct'] - 0.5) * 0.05,
            'tov_pct': 0.135 - (proj['win_pct'] - 0.5) * 0.03,
            'ft_rate': 0.25 + (proj['off_rating'] - 115) * 0.005,
            
            # Estatísticas básicas
            'fg_pct': 0.47 + (proj['off_rating'] - 115) * 0.002,
            'fg3_pct': 0.36 + (proj['off_rating'] - 115) * 0.002,
            'ft_pct': 0.78,
            
            # Calculados
            'possessions': 100,
            'last_10_games': [1 if i < proj['win_pct'] * 10 else 0 for i in range(10)],
            'home_record': f"{int(41 * proj['win_pct'])}-{41 - int(41 * proj['win_pct'])}",
            'away_record': f"{int(41 * proj['win_pct'])}-{41 - int(41 * proj['win_pct'])}",
            'vs_east': proj['win_pct'],
            'vs_west': proj['win_pct'],
            'days_rest': 2,
            'injury_impact': np.random.uniform(0.95, 1.05)
        }
        
        # Adicionar métricas de tendência
        stats['momentum'] = np.random.uniform(-0.1, 0.1) + (proj['win_pct'] - 0.5) * 0.2
        stats['consistency'] = np.random.uniform(0.7, 0.95) - (1 - proj['win_pct']) * 0.1
        
        return stats
    
    def _get_backup_stats(self, team_name: str) -> Dict:
        """Estatísticas de backup caso falhe tudo"""
        return {
            'name': team_name,
            'season': self.season,
            'win_pct': 0.5,
            'off_rating': 115.0,
            'def_rating': 115.0,
            'net_rating': 0.0,
            'pace': 100.0,
            'points': 115.0,
            'opp_points': 115.0,
            'effective_fg_pct': 0.545,
            'true_shooting_pct': 0.580,
            'fg_pct': 0.47,
            'fg3_pct': 0.36,
            'ft_pct': 0.78
        }
    
    def get_conference(self, team_name: str) -> str:
        """Retorna a conferência de um time"""
        east_teams = [t for t in self.get_all_teams() if any(x in t for x in ['Celtics', 'Knicks', '76ers', 'Nets', 'Raptors', 
                                                                             'Bulls', 'Cavaliers', 'Pistons', 'Pacers', 
                                                                             'Bucks', 'Hawks', 'Hornets', 'Heat', 'Magic', 'Wizards'])]
        return "Leste" if team_name in east_teams else "Oeste"

# ============================================================================
# 2. MACHINE LEARNING PARA PREVISÕES
# ============================================================================

class NBAPredictiveML:
    """Sistema de Machine Learning para previsões NBA"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'off_rating_diff', 'def_rating_diff', 'net_rating_diff',
            'win_pct_diff', 'home_advantage', 'rest_advantage',
            'momentum_diff', 'consistency_avg', 'pace_diff'
        ]
        self.is_trained = False
        
    def train_model(self, training_data: List[Dict]):
        """Treina o modelo com dados históricos"""
        if not training_data:
            return
        
        X = []
        y = []
        
        for game in training_data:
            features = self._extract_features(game)
            if features:
                X.append(features)
                y.append(1 if game['home_score'] > game['away_score'] else 0)
        
        if X and y:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
    
    def predict_matchup(self, home_stats: Dict, away_stats: Dict, is_home: bool = True) -> Dict:
        """Prevê resultado de um confronto"""
        
        # Extrair features
        features = self._calculate_features(home_stats, away_stats, is_home)
        
        if self.is_trained:
            # Usar modelo ML
            features_scaled = self.scaler.transform([features])
            win_prob = self.model.predict_proba(features_scaled)[0][1]
        else:
            # Fallback: modelo estatístico
            win_prob = self._statistical_prediction(home_stats, away_stats, is_home)
        
        # Calcular margem e total
        margin, total = self._calculate_margin_and_total(home_stats, away_stats, is_home)
        
        return {
            'win_probability': win_prob,
            'predicted_margin': margin,
            'predicted_total': total,
            'confidence': self._calculate_confidence(home_stats, away_stats, win_prob),
            'key_factors': self._analyze_key_factors(home_stats, away_stats)
        }
    
    def _calculate_features(self, home: Dict, away: Dict, is_home: bool) -> List[float]:
        """Calcula features para o modelo"""
        
        home_adv = 3.5 if is_home else -3.5
        rest_adv = home.get('days_rest', 2) - away.get('days_rest', 2)
        
        features = [
            home['off_rating'] - away['off_rating'],  # off_rating_diff
            away['def_rating'] - home['def_rating'],  # def_rating_diff (invertido)
            home['net_rating'] - away['net_rating'],  # net_rating_diff
            home['win_pct'] - away['win_pct'],        # win_pct_diff
            home_adv,                                 # home_advantage
            min(rest_adv, 3),                         # rest_advantage (capped)
            home.get('momentum', 0) - away.get('momentum', 0),  # momentum_diff
            (home.get('consistency', 0.8) + away.get('consistency', 0.8)) / 2,  # consistency_avg
            home['pace'] - away['pace']              # pace_diff
        ]
        
        return features
    
    def _statistical_prediction(self, home: Dict, away: Dict, is_home: bool) -> float:
        """Previsão estatística baseada em múltiplos fatores"""
        
        # Peso dos fatores
        weights = {
            'net_rating': 0.30,
            'win_pct': 0.25,
            'home_advantage': 0.20,
            'efficiency': 0.15,
            'momentum': 0.10
        }
        
        # Calcular scores
        net_score = (home['net_rating'] - away['net_rating']) * 0.1
        win_pct_score = (home['win_pct'] - away['win_pct']) * 20
        home_score = 3.5 if is_home else -3.5
        eff_score = ((home['off_rating'] - away['def_rating']) - (away['off_rating'] - home['def_rating'])) * 0.05
        momentum_score = (home.get('momentum', 0) - away.get('momentum', 0)) * 10
        
        # Score total
        total_score = (
            net_score * weights['net_rating'] +
            win_pct_score * weights['win_pct'] +
            home_score * weights['home_advantage'] +
            eff_score * weights['efficiency'] +
            momentum_score * weights['momentum']
        )
        
        # Converter para probabilidade
        win_prob = 1 / (1 + np.exp(-total_score * 0.15))
        
        return win_prob
    
    def _calculate_margin_and_total(self, home: Dict, away: Dict, is_home: bool) -> Tuple[float, float]:
        """Calcula margem e total esperados"""
        
        # Pontos esperados
        home_points = home['points'] + (3.5 if is_home else -1.5)
        away_points = away['points'] + (-1.5 if is_home else 3.5)
        
        # Ajustar por diferença de qualidade
        quality_diff = (home['net_rating'] - away['net_rating']) * 0.1
        home_points += quality_diff
        away_points -= quality_diff
        
        margin = home_points - away_points
        total = home_points + away_points
        
        return margin, total
    
    def _calculate_confidence(self, home: Dict, away: Dict, win_prob: float) -> float:
        """Calcula confiança da previsão"""
        
        # Fatores de confiança
        consistency = (home.get('consistency', 0.8) + away.get('consistency', 0.8)) / 2
        data_quality = 0.9  # Qualidade dos dados (0-1)
        
        # Confiança baseada na probabilidade e consistência
        if win_prob > 0.7 or win_prob < 0.3:
            base_confidence = 0.85
        elif win_prob > 0.6 or win_prob < 0.4:
            base_confidence = 0.75
        else:
            base_confidence = 0.65
        
        confidence = base_confidence * consistency * data_quality
        return min(max(confidence, 0.5), 0.95)
    
    def _analyze_key_factors(self, home: Dict, away: Dict) -> List[Dict]:
        """Analisa fatores-chave do confronto"""
        
        factors = [
            {
                'name': 'Eficiência Ofensiva',
                'home_value': home['off_rating'],
                'away_value': away['off_rating'],
                'advantage': 'home' if home['off_rating'] > away['off_rating'] else 'away',
                'impact': abs(home['off_rating'] - away['off_rating']) * 0.1
            },
            {
                'name': 'Eficiência Defensiva',
                'home_value': home['def_rating'],
                'away_value': away['def_rating'],
                'advantage': 'home' if home['def_rating'] < away['def_rating'] else 'away',
                'impact': abs(home['def_rating'] - away['def_rating']) * 0.1
            },
            {
                'name': 'Ritmo de Jogo',
                'home_value': home['pace'],
                'away_value': away['pace'],
                'advantage': 'home' if home['pace'] > away['pace'] else 'away',
                'impact': abs(home['pace'] - away['pace']) * 0.05
            },
            {
                'name': 'Arremesso Efetivo',
                'home_value': home['effective_fg_pct'],
                'away_value': away['effective_fg_pct'],
                'advantage': 'home' if home['effective_fg_pct'] > away['effective_fg_pct'] else 'away',
                'impact': abs(home['effective_fg_pct'] - away['effective_fg_pct']) * 100
            },
            {
                'name': 'Consistência',
                'home_value': home.get('consistency', 0.8),
                'away_value': away.get('consistency', 0.8),
                'advantage': 'home' if home.get('consistency', 0.8) > away.get('consistency', 0.8) else 'away',
                'impact': abs(home.get('consistency', 0.8) - away.get('consistency', 0.8)) * 10
            }
        ]
        
        return factors

# ============================================================================
# 3. SISTEMA DE SIMULAÇÃO AVANÇADO
# ============================================================================

class AdvancedNBASimulator:
    """Sistema avançado de simulação"""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        
    def monte_carlo_simulation(self, home_stats: Dict, away_stats: Dict, is_home: bool = True) -> Dict:
        """Simulação Monte Carlo avançada"""
        
        np.random.seed(42)  # Para reproducibilidade
        
        results = {
            'home_wins': 0,
            'away_wins': 0,
            'margins': [],
            'totals': [],
            'home_scores': [],
            'away_scores': [],
            'overtime_games': 0
        }
        
        for _ in range(self.n_simulations):
            # Simular scores com distribuição baseada em estatísticas
            home_score = self._simulate_team_score(home_stats, away_stats, is_home, is_home_team=True)
            away_score = self._simulate_team_score(away_stats, home_stats, not is_home, is_home_team=False)
            
            # Verificar overtime
            if abs(home_score - away_score) <= 2:
                # Simular overtime
                ot_home = np.random.normal(12, 4)
                ot_away = np.random.normal(12, 4)
                home_score += ot_home
                away_score += ot_away
                results['overtime_games'] += 1
            
            results['home_scores'].append(home_score)
            results['away_scores'].append(away_score)
            results['totals'].append(home_score + away_score)
            results['margins'].append(home_score - away_score)
            
            if home_score > away_score:
                results['home_wins'] += 1
            else:
                results['away_wins'] += 1
        
        # Calcular estatísticas
        win_prob = results['home_wins'] / self.n_simulations
        avg_margin = np.mean(results['margins'])
        avg_total = np.mean(results['totals'])
        
        # Calcular intervalos de confiança
        margin_ci = np.percentile(results['margins'], [2.5, 97.5])
        total_ci = np.percentile(results['totals'], [2.5, 97.5])
        
        # Calcular probabilidades over/under
        over_under = self._calculate_over_under_probabilities(results['totals'])
        
        return {
            'win_probability': win_prob,
            'predicted_margin': avg_margin,
            'predicted_total': avg_total,
            'margin_confidence_interval': margin_ci.tolist(),
            'total_confidence_interval': total_ci.tolist(),
            'over_under_probabilities': over_under,
            'overtime_probability': results['overtime_games'] / self.n_simulations,
            'simulation_stats': {
                'home_score_mean': np.mean(results['home_scores']),
                'home_score_std': np.std(results['home_scores']),
                'away_score_mean': np.mean(results['away_scores']),
                'away_score_std': np.std(results['away_scores'])
            }
        }
    
    def _simulate_team_score(self, team_stats: Dict, opponent_stats: Dict, is_home: bool, is_home_team: bool) -> float:
        """Simula pontuação de um time"""
        
        # Pontos esperados base
        expected_points = team_stats['points']
        
        # Ajustes
        if is_home:
            expected_points += 3.5 if is_home_team else -1.5
        
        # Ajuste por qualidade do oponente
        quality_adjustment = (team_stats['off_rating'] - opponent_stats['def_rating']) * 0.1
        expected_points += quality_adjustment
        
        # Variação baseada na consistência
        consistency = team_stats.get('consistency', 0.8)
        std_dev = 10 * (1.2 - consistency)  # Times mais consistentes têm menos variação
        
        # Simular score
        simulated_score = np.random.normal(expected_points, std_dev)
        
        # Garantir limites razoáveis
        simulated_score = max(80, min(150, simulated_score))
        
        return simulated_score
    
    def _calculate_over_under_probabilities(self, totals: List[float]) -> Dict:
        """Calcula probabilidades over/under"""
        
        probs = {}
        total_array = np.array(totals)
        
        # Linhas comuns
        lines = [200, 205, 210, 212.5, 215, 217.5, 220, 222.5, 225, 227.5, 230]
        
        for line in lines:
            over_prob = np.mean(total_array > line)
            under_prob = np.mean(total_array < line)
            probs[f'over_{line}'] = over_prob
            probs[f'under_{line}'] = under_prob
        
        return probs

# ============================================================================
# 4. SISTEMA PRINCIPAL NBA ULTIMATE
# ============================================================================

class NBAUltimateAnalytics2026:
    """Sistema principal de análise"""
    
    def __init__(self):
        self.data_engine = NBA2026DataEngine()
        self.ml_predictor = NBAPredictiveML()
        self.simulator = AdvancedNBASimulator(n_simulations=10000)
        
    def analyze_matchup(self, home_team: str, away_team: str, venue: str = 'home', 
                       home_days_rest: int = 2, away_days_rest: int = 2) -> Dict:
        """Executa análise completa de um confronto"""
        
        with st.spinner(f"🔄 Analisando {home_team} vs {away_team}..."):
            # Buscar dados
            home_stats = self.data_engine.get_team_stats(home_team)
            away_stats = self.data_engine.get_team_stats(away_team)
            
            # Atualizar dias de descanso
            home_stats['days_rest'] = home_days_rest
            away_stats['days_rest'] = away_days_rest
            
            is_home = (venue == 'home')
            
            # 1. Previsão ML
            ml_prediction = self.ml_predictor.predict_matchup(home_stats, away_stats, is_home)
            
            # 2. Simulação Monte Carlo
            simulation_results = self.simulator.monte_carlo_simulation(home_stats, away_stats, is_home)
            
            # 3. Análise dos 4 Fatores
            four_factors = self._analyze_four_factors(home_stats, away_stats)
            
            # 4. Análise de Matchups
            matchup_analysis = self._analyze_matchups(home_stats, away_stats)
            
            # 5. Combinação de modelos
            final_prediction = self._combine_predictions(ml_prediction, simulation_results)
            
            # 6. Gerar insights
            insights = self._generate_insights(home_stats, away_stats, final_prediction, matchup_analysis)
            
            # 7. Obter conferências
            home_conf = self.data_engine.get_conference(home_team)
            away_conf = self.data_engine.get_conference(away_team)
            
            analysis = {
                'matchup': f"{home_team} vs {away_team}",
                'home_team': home_team,
                'away_team': away_team,
                'venue': venue,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'season': '2025-26',
                
                # Conferências
                'home_conference': home_conf,
                'away_conference': away_conf,
                
                # Previsões
                'final_prediction': final_prediction,
                'ml_prediction': ml_prediction,
                'simulation_results': simulation_results,
                
                # Análises
                'four_factors_analysis': four_factors,
                'matchup_analysis': matchup_analysis,
                'insights': insights,
                
                # Dados
                'home_stats': home_stats,
                'away_stats': away_stats,
                
                # Metadados
                'analysis_id': f"{home_team.replace(' ', '_')}_{away_team.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'confidence_score': final_prediction['confidence'] * 100
            }
            
            # Salvar análise
            self._save_analysis(analysis)
            
            return analysis
    
    def _analyze_four_factors(self, home: Dict, away: Dict) -> Dict:
        """Analisa os Quatro Fatores de Dean Oliver"""
        
        factors = {
            'shooting': {
                'home': home['effective_fg_pct'],
                'away': away['effective_fg_pct'],
                'advantage': 'home' if home['effective_fg_pct'] > away['effective_fg_pct'] else 'away',
                'impact': 'Alta' if abs(home['effective_fg_pct'] - away['effective_fg_pct']) > 0.02 else 'Média'
            },
            'turnovers': {
                'home': 1 - home.get('tov_pct', 0.135),
                'away': 1 - away.get('tov_pct', 0.135),
                'advantage': 'home' if home.get('tov_pct', 0.135) < away.get('tov_pct', 0.135) else 'away',
                'impact': 'Alta' if abs(home.get('tov_pct', 0.135) - away.get('tov_pct', 0.135)) > 0.02 else 'Média'
            },
            'rebounding': {
                'home': home.get('off_reb_pct', 0.25),
                'away': away.get('off_reb_pct', 0.25),
                'advantage': 'home' if home.get('off_reb_pct', 0.25) > away.get('off_reb_pct', 0.25) else 'away',
                'impact': 'Média'
            },
            'free_throws': {
                'home': home.get('ft_rate', 0.25),
                'away': away.get('ft_rate', 0.25),
                'advantage': 'home' if home.get('ft_rate', 0.25) > away.get('ft_rate', 0.25) else 'away',
                'impact': 'Baixa'
            }
        }
        
        # Contar vantagens
        home_advantages = sum(1 for f in factors.values() if f['advantage'] == 'home')
        away_advantages = 4 - home_advantages
        
        factors['summary'] = {
            'home_advantages': home_advantages,
            'away_advantages': away_advantages,
            'dominant_team': 'home' if home_advantages >= 3 else 'away' if away_advantages >= 3 else 'balanced'
        }
        
        return factors
    
    def _analyze_matchups(self, home: Dict, away: Dict) -> Dict:
        """Analisa matchups específicos"""
        
        analysis = {
            'offensive_matchup': {
                'description': f"{home['name']} (ORtg: {home['off_rating']:.1f}) vs {away['name']} (DRtg: {away['def_rating']:.1f})",
                'advantage': 'home' if home['off_rating'] > away['def_rating'] else 'away',
                'magnitude': abs(home['off_rating'] - away['def_rating'])
            },
            'defensive_matchup': {
                'description': f"{away['name']} (ORtg: {away['off_rating']:.1f}) vs {home['name']} (DRtg: {home['def_rating']:.1f})",
                'advantage': 'home' if away['off_rating'] < home['def_rating'] else 'away',
                'magnitude': abs(away['off_rating'] - home['def_rating'])
            },
            'pace_analysis': {
                'description': f"Ritmo: {home['name']} ({home['pace']:.1f}) vs {away['name']} ({away['pace']:.1f})",
                'game_type': 'Rápido' if (home['pace'] + away['pace']) / 2 > 102 else 
                            'Lento' if (home['pace'] + away['pace']) / 2 < 98 else 'Médio',
                'advantage': 'home' if home['pace'] > away['pace'] else 'away'
            }
        }
        
        return analysis
    
    def _combine_predictions(self, ml_prediction: Dict, simulation: Dict) -> Dict:
        """Combina previsões de múltiplos modelos"""
        
        # Média ponderada
        ml_weight = 0.4
        sim_weight = 0.6
        
        combined_win_prob = (
            ml_prediction['win_probability'] * ml_weight +
            simulation['win_probability'] * sim_weight
        )
        
        # Margem combinada
        combined_margin = (
            ml_prediction['predicted_margin'] * 0.3 +
            simulation['predicted_margin'] * 0.7
        )
        
        # Total combinado
        combined_total = simulation['predicted_total']  # Usar apenas simulação para total
        
        # Confiança combinada
        confidence = (ml_prediction['confidence'] + 0.8) / 2  # Ponderar
        
        return {
            'win_probability': combined_win_prob,
            'predicted_margin': combined_margin,
            'predicted_total': combined_total,
            'confidence': confidence,
            'model_weights': {'ml': ml_weight, 'simulation': sim_weight}
        }
    
    def _generate_insights(self, home: Dict, away: Dict, prediction: Dict, matchup: Dict) -> List[str]:
        """Gera insights estratégicos"""
        
        insights = []
        
        # Insight 1: Probabilidade
        win_prob = prediction['win_probability']
        if win_prob > 0.65:
            insights.append(f"**Forte favoritismo** para {'casa' if win_prob > 0.5 else 'visitante'}")
        elif win_prob > 0.55:
            insights.append(f"**Leve favoritismo** para {'casa' if win_prob > 0.5 else 'visitante'}")
        else:
            insights.append("**Jogo equilibrado** - difícil previsão")
        
        # Insight 2: Margem
        margin = prediction['predicted_margin']
        if abs(margin) > 8:
            insights.append(f"Expectativa de jogo **{'unilateral' if abs(margin) > 10 else 'com boa margem'}")
        elif abs(margin) > 4:
            insights.append("Expectativa de jogo **competitivo** com pequena margem")
        else:
            insights.append("**Jogo apertado** - pode ser decidido nos minutos finais")
        
        # Insight 3: Total
        total = prediction['predicted_total']
        if total > 225:
            insights.append("**Alto scoring esperado** - aposte no over")
        elif total > 215:
            insights.append("**Scoring médio-alto** - jogo ofensivo")
        else:
            insights.append("**Jogo defensivo** - aposte no under")
        
        # Insight 4: Matchup ofensivo
        off_matchup = matchup['offensive_matchup']
        if off_matchup['magnitude'] > 5:
            insights.append(f"**Vantagem ofensiva significativa** para {off_matchup['advantage']}")
        
        # Insight 5: Ritmo
        pace_type = matchup['pace_analysis']['game_type']
        insights.append(f"Expectativa de jogo **{pace_type.lower()}**")
        
        # Insight 6: Confiança
        confidence = prediction['confidence']
        if confidence > 0.8:
            insights.append("**Alta confiança** na previsão")
        elif confidence > 0.7:
            insights.append("**Confiança moderada** na previsão")
        else:
            insights.append("**Baixa confiança** - muitos fatores imprevisíveis")
        
        return insights
    
    def _save_analysis(self, analysis: Dict):
        """Salva análise no banco de dados"""
        try:
            cursor = self.data_engine.cursor
            cursor.execute('''
                INSERT INTO saved_analyses 
                (timestamp, matchup, prediction_data, accuracy_score, features_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                analysis['matchup'],
                json.dumps(analysis['final_prediction']),
                analysis['confidence_score'],
                json.dumps(list(self.ml_predictor.feature_names))
            ))
            self.data_engine.conn.commit()
        except:
            pass

# ============================================================================
# 5. INTERFACE STREAMLIT PROFISSIONAL (COM CORREÇÕES)
# ============================================================================

class NBAUltimateUI:
    """Interface do usuário profissional"""
    
    def __init__(self):
        self.analytics_system = NBAUltimateAnalytics2026()
        self.data_engine = self.analytics_system.data_engine
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Inicializa estado da sessão"""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'favorite_teams' not in st.session_state:
            st.session_state.favorite_teams = ["Boston Celtics", "Golden State Warriors", "Los Angeles Lakers"]
        if 'home_team' not in st.session_state:
            st.session_state.home_team = "Boston Celtics"
        if 'away_team' not in st.session_state:
            st.session_state.away_team = "Golden State Warriors"
    
    def render_sidebar(self):
        """Renderiza sidebar profissional"""
        with st.sidebar:
            # Header
            st.markdown('<div class="sidebar-header"><h3>🏀 NBA ULTIMATE ANALYTICS</h3><p>Temporada 2025-26</p></div>', 
                       unsafe_allow_html=True)
            
            # Status
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Temporada", "2025-26", "Nova")
            with col2:
                st.metric("Times", "30", "")
            
            st.markdown("---")
            
            # ================================================================
            # CORREÇÃO: SELEÇÃO LIVRE DE TIMES (CASA/VISITANTE)
            # ================================================================
            st.subheader("🔍 CONFIGURAR ANÁLISE")
            
            all_teams = self.data_engine.get_all_teams()
            
            # Dividir em conferências apenas para referência
            east_teams = [t for t in all_teams if t in [
                "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
                "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers",
                "Miami Heat", "Milwaukee Bucks", "New York Knicks", "Orlando Magic",
                "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards"
            ]]
            
            west_teams = [t for t in all_teams if t in [
                "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets",
                "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Minnesota Timberwolves",
                "New Orleans Pelicans", "Oklahoma City Thunder", "Phoenix Suns", "Portland Trail Blazers",
                "Sacramento Kings", "San Antonio Spurs", "Utah Jazz"
            ]]
            
            # Abas para seleção de times
            home_tab, away_tab = st.tabs(["🏠 TIME DA CASA", "✈️ TIME VISITANTE"])
            
            with home_tab:
                st.markdown("**Selecione qualquer time como Time da Casa:**")
                home_team = st.selectbox(
                    "Time da Casa:",
                    all_teams,
                    index=all_teams.index(st.session_state.home_team) if st.session_state.home_team in all_teams else 0,
                    key="home_team_select",
                    help="Qualquer time pode ser selecionado como time da casa"
                )
                
                # Mostrar conferência do time selecionado
                home_conf = self.data_engine.get_conference(home_team)
                st.caption(f"📋 Conferência: **{home_conf}**")
            
            with away_tab:
                st.markdown("**Selecione qualquer time como Time Visitante:**")
                
                # Filtrar para remover o time da casa das opções
                away_options = [team for team in all_teams if team != home_team]
                
                # Garantir que temos uma opção válida
                if not away_options:
                    away_options = all_teams.copy()
                
                away_team = st.selectbox(
                    "Time Visitante:",
                    away_options,
                    index=0 if away_options else 0,
                    key="away_team_select",
                    help="Qualquer time pode ser selecionado como time visitante"
                )
                
                # Mostrar conferência do time selecionado
                away_conf = self.data_engine.get_conference(away_team)
                st.caption(f"📋 Conferência: **{away_conf}**")
            
            # Validação de seleção
            if home_team == away_team:
                st.markdown('<div class="validation-warning">⚠️ <strong>Atenção:</strong> Selecione times diferentes para casa e visitante!</div>', 
                           unsafe_allow_html=True)
            
            # Informação do matchup
            st.markdown("---")
            st.markdown("**🤝 MATCHUP SELECIONADO:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #17408B; color: white; border-radius: 10px;">
                    <h4>🏠 CASA</h4>
                    <h3>{home_team}</h3>
                    <p>{home_conf}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #C9082A; color: white; border-radius: 10px;">
                    <h4>✈️ VISITANTE</h4>
                    <h3>{away_team}</h3>
                    <p>{away_conf}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Verificar se é um jogo intra-conferência
            if home_conf == away_conf:
                st.info(f"⚔️ **Jogo Intra-Conferência ({home_conf})**")
            else:
                st.info(f"🌉 **Jogo Inter-Conferência ({home_conf} vs {away_conf})**")
            
            # Configurações avançadas
            with st.expander("⚙️ CONFIGURAÇÕES AVANÇADAS"):
                venue_options = {
                    "🏠 Casa (Vantagem +3.5 pts)": "home",
                    "✈️ Visitante (Vantagem +3.5 pts)": "away", 
                    "🏆 Neutro (Sem vantagem)": "neutral"
                }
                
                venue_display = st.radio(
                    "Local do Jogo:",
                    list(venue_options.keys()),
                    horizontal=True
                )
                
                venue_type = venue_options[venue_display]
                
                col1, col2 = st.columns(2)
                with col1:
                    home_rest = st.slider("Descanso Casa:", 0, 7, 2, help="Dias desde o último jogo")
                with col2:
                    away_rest = st.slider("Descanso Visitante:", 0, 7, 2, help="Dias desde o último jogo")
                
                # Fatores adicionais
                st.markdown("**🎯 Fatores Adicionais:**")
                injury_impact = st.slider("Impacto de Lesões:", 0.8, 1.2, 1.0, 0.05)
                fatigue_factor = st.slider("Fator de Fadiga:", 0.9, 1.1, 1.0, 0.05)
            
            st.markdown("---")
            
            # Botão de análise
            analyze_button = st.button("🚀 EXECUTAR ANÁLISE COMPLETA", 
                                     type="primary", 
                                     use_container_width=True,
                                     disabled=(home_team == away_team))
            
            if analyze_button:
                # Salvar seleção atual
                st.session_state.home_team = home_team
                st.session_state.away_team = away_team
                
                with st.spinner("Processando análise avançada..."):
                    analysis = self.analytics_system.analyze_matchup(
                        home_team, away_team, venue_type, home_rest, away_rest
                    )
                    st.session_state.current_analysis = analysis
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'matchup': analysis['matchup'],
                        'home_team': home_team,
                        'away_team': away_team,
                        'win_prob': analysis['final_prediction']['win_probability']
                    })
                st.rerun()
            
            # Histórico
            st.markdown("---")
            st.subheader("📊 HISTÓRICO")
            
            if st.session_state.analysis_history:
                for i, hist in enumerate(st.session_state.analysis_history[-5:]):
                    with st.expander(f"{hist['matchup']} - {hist['win_prob']:.1%}", expanded=False):
                        st.write(f"**Data:** {hist['timestamp'].strftime('%d/%m %H:%M')}")
                        st.write(f"**Casa:** {hist['home_team']}")
                        st.write(f"**Visitante:** {hist['away_team']}")
                        
                        # Botão para recarregar análise
                        if st.button(f"↻ Recarregar", key=f"reload_{i}"):
                            st.session_state.home_team = hist['home_team']
                            st.session_state.away_team = hist['away_team']
                            st.rerun()
            else:
                st.info("Nenhuma análise realizada ainda")
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
                <p>NBA Ultimate Analytics v3.1</p>
                <p>© 2025 - Sistema Profissional</p>
                <p>Seleção Livre de Times ✅</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_dashboard(self, analysis: Dict):
        """Renderiza dashboard completo"""
        
        if not analysis:
            st.warning("⚠️ Execute uma análise primeiro")
            return
        
        # Header da análise
        st.markdown(f"""
        <div class="fade-in">
            <h2 style="text-align: center; margin-bottom: 30px;">
                🏆 {analysis['matchup']} • {analysis['season']}
            </h2>
            <p style="text-align: center; color: #666; margin-bottom: 40px;">
                {analysis['home_team']} ({analysis['home_conference']}) vs {analysis['away_team']} ({analysis['away_conference']}) • 
                Análise realizada em {analysis['date']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Abas principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 RESUMO", 
            "🎲 SIMULAÇÃO", 
            "📈 ESTATÍSTICAS", 
            "🔍 ANÁLISE", 
            "💡 INSIGHTS"
        ])
        
        with tab1:
            self._render_summary_tab(analysis)
        
        with tab2:
            self._render_simulation_tab(analysis)
        
        with tab3:
            self._render_stats_tab(analysis)
        
        with tab4:
            self._render_analysis_tab(analysis)
        
        with tab5:
            self._render_insights_tab(analysis)
    
    def _render_summary_tab(self, analysis: Dict):
        """Renderiza aba de resumo"""
        
        prediction = analysis['final_prediction']
        home_team = analysis['home_team']
        away_team = analysis['away_team']
        home_conf = analysis['home_conference']
        away_conf = analysis['away_conference']
        
        # Informação do matchup
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #17408B 0%, #1e4f9e 100%); 
                        padding: 20px; border-radius: 15px; color: white;">
                <h3>🏠 {home_team}</h3>
                <p>Conferência: {home_conf}</p>
                <p>Recorde: {analysis['home_stats'].get('home_record', '0-0')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #C9082A 0%, #d91a3a 100%); 
                        padding: 20px; border-radius: 15px; color: white;">
                <h3>✈️ {away_team}</h3>
                <p>Conferência: {away_conf}</p>
                <p>Recorde: {analysis['away_stats'].get('away_record', '0-0')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Cards principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            win_prob = prediction['win_probability']
            favorite = home_team if win_prob > 0.5 else away_team
            confidence = prediction['confidence']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 PROBABILIDADE</h3>
                <h1 style="color: {'#28a745' if win_prob > 0.5 else '#dc3545'}; font-size: 3rem;">
                    {win_prob:.1%}
                </h1>
                <p><strong>Favorito:</strong> {favorite}</p>
                <p><strong>Confiança:</strong> {confidence:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            margin = prediction['predicted_margin']
            abs_margin = abs(margin)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 MARGEM ESPERADA</h3>
                <h1 style="color: {'#28a745' if margin > 0 else '#dc3545'}; font-size: 3rem;">
                    {abs_margin:.1f} pts
                </h1>
                <p><strong>Líder:</strong> {home_team if margin > 0 else away_team}</p>
                <p><strong>Diferencial:</strong> {margin:+.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total = prediction['predicted_total']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 TOTAL DE PONTOS</h3>
                <h1 style="color: #1d428a; font-size: 3rem;">{total:.1f}</h1>
                <p><strong>Tipo de Jogo:</strong> {'Alto Scoring' if total > 220 else 'Médio' if total > 210 else 'Baixo'}</p>
                <p><strong>Over/Under:</strong> {'OVER' if total > 215 else 'UNDER'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            four_factors = analysis['four_factors_analysis']['summary']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚖️ 4 FATORES</h3>
                <h1 style="color: #6f42c1; font-size: 3rem;">
                    {four_factors['home_advantages']}-{four_factors['away_advantages']}
                </h1>
                <p><strong>Vantagem:</strong> {four_factors['dominant_team'].upper()}</p>
                <p><strong>Equilíbrio:</strong> {'Desequilibrado' if four_factors['dominant_team'] != 'balanced' else 'Equilibrado'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de probabilidade
        st.markdown("---")
        st.subheader("📊 DISTRIBUIÇÃO DA PROBABILIDADE")
        
        fig = go.Figure()
        
        # Adicionar barras
        fig.add_trace(go.Bar(
            x=[home_team, away_team],
            y=[win_prob, 1 - win_prob],
            marker_color=['#17408B', '#C9082A'],
            text=[f'{win_prob:.1%}', f'{1-win_prob:.1%}'],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f'Probabilidade de Vitória - {home_team} vs {away_team}',
            yaxis_title='Probabilidade',
            yaxis_tickformat='.0%',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights rápidos
        st.markdown("---")
        st.subheader("⚡ INSIGHTS RÁPIDOS")
        
        for insight in analysis['insights'][:3]:
            st.info(f"• {insight}")
    
    def _render_simulation_tab(self, analysis: Dict):
        """Renderiza aba de simulação"""
        
        sim_results = analysis['simulation_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição das margens
            st.subheader("📈 Distribuição da Margem")
            
            # Criar dados simulados para visualização
            np.random.seed(42)
            simulated_margins = np.random.normal(
                sim_results['predicted_margin'],
                sim_results['simulation_stats']['home_score_std'] + sim_results['simulation_stats']['away_score_std'],
                10000
            )
            
            fig = px.histogram(
                x=simulated_margins,
                nbins=50,
                title='Distribuição da Margem de Vitória',
                labels={'x': 'Margem de Pontos', 'y': 'Frequência'},
                color_discrete_sequence=['#17408B']
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Empate")
            fig.add_vline(
                x=sim_results['predicted_margin'],
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Média: {sim_results['predicted_margin']:.1f}"
            )
            
            # Adicionar intervalo de confiança
            ci = sim_results['margin_confidence_interval']
            fig.add_vrect(
                x0=ci[0], x1=ci[1],
                fillcolor="rgba(23, 64, 139, 0.2)",
                line_width=0,
                annotation_text=f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Probabilidades Over/Under
            st.subheader("💰 Probabilidades Over/Under")
            
            over_under = sim_results['over_under_probabilities']
            lines = [210, 215, 220, 225, 230]
            
            over_probs = [over_under.get(f'over_{line}', 0.5) for line in lines]
            under_probs = [over_under.get(f'under_{line}', 0.5) for line in lines]
            
            fig = go.Figure(data=[
                go.Bar(name='Over', x=lines, y=over_probs, marker_color='#28a745'),
                go.Bar(name='Under', x=lines, y=under_probs, marker_color='#dc3545')
            ])
            
            fig.update_layout(
                barmode='group',
                title='Probabilidade Over/Under por Linha',
                xaxis_title='Linha de Total',
                yaxis_title='Probabilidade',
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estatísticas da simulação
            st.markdown("**📊 ESTATÍSTICAS DA SIMULAÇÃO:**")
            
            sim_stats = [
                ("Simulações", f"{sim_results.get('n_simulations', 10000):,}"),
                ("Vitórias Casa", f"{sim_results['win_probability']:.1%}"),
                ("Média Casa", f"{sim_results['simulation_stats']['home_score_mean']:.1f}"),
                ("Média Visitante", f"{sim_results['simulation_stats']['away_score_mean']:.1f}"),
                ("Prob. Overtime", f"{sim_results.get('overtime_probability', 0):.1%}"),
                ("Confiança 95%", f"[{sim_results['margin_confidence_interval'][0]:.1f}, {sim_results['margin_confidence_interval'][1]:.1f}]")
            ]
            
            for stat_name, stat_value in sim_stats:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.text(stat_name)
                with col_b:
                    st.text(stat_value)
    
    def _render_stats_tab(self, analysis: Dict):
        """Renderiza aba de estatísticas"""
        
        home_stats = analysis['home_stats']
        away_stats = analysis['away_stats']
        
        # Métricas comparativas
        st.subheader("📊 COMPARAÇÃO DE ESTATÍSTICAS")
        
        metrics_to_compare = [
            ('Offensive Rating', 'off_rating', 'higher'),
            ('Defensive Rating', 'def_rating', 'lower'),
            ('Net Rating', 'net_rating', 'higher'),
            ('Win %', 'win_pct', 'higher'),
            ('Pace', 'pace', 'context'),
            ('Effective FG%', 'effective_fg_pct', 'higher'),
            ('True Shooting%', 'true_shooting_pct', 'higher')
        ]
        
        # Criar DataFrame comparativo
        comparison_data = []
        for metric_name, metric_key, better in metrics_to_compare:
            home_val = home_stats.get(metric_key, 0)
            away_val = away_stats.get(metric_key, 0)
            
            if better == 'higher':
                advantage = 'home' if home_val > away_val else 'away'
            elif better == 'lower':
                advantage = 'home' if home_val < away_val else 'away'
            else:
                advantage = 'neutral'
            
            comparison_data.append({
                'Metric': metric_name,
                analysis['home_team']: f"{home_val:.2f}" if isinstance(home_val, float) else str(home_val),
                analysis['away_team']: f"{away_val:.2f}" if isinstance(away_val, float) else str(away_val),
                'Advantage': advantage,
                'Difference': f"{home_val - away_val:+.2f}" if isinstance(home_val, (int, float)) else "N/A"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Função para colorir vantagens
        def color_advantage(val):
            if val == 'home':
                return 'background-color: rgba(23, 64, 139, 0.2);'
            elif val == 'away':
                return 'background-color: rgba(201, 8, 42, 0.2);'
            return ''
        
        # Aplicar estilo
        styled_df = df_comparison.style.applymap(
            color_advantage, 
            subset=['Advantage']
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Gráfico de radar
        st.markdown("---")
        st.subheader("📈 PERFIL DAS EQUIPES (RADAR)")
        
        categories = ['Offense', 'Defense', 'Net Rating', 'Shooting', 'Pace', 'Consistency']
        
        # Normalizar valores para escala 0-100
        def normalize(value, min_val, max_val):
            return ((value - min_val) / (max_val - min_val)) * 100 if max_val > min_val else 50
        
        values_home = [
            normalize(home_stats['off_rating'], 105, 125),
            normalize(130 - home_stats['def_rating'], 105, 125),  # Inverter para defesa (menor é melhor)
            normalize(home_stats['net_rating'] + 20, -10, 30),
            normalize(home_stats['effective_fg_pct'] * 100, 45, 60),
            normalize(home_stats['pace'], 95, 105),
            normalize(home_stats.get('consistency', 0.8) * 100, 50, 100)
        ]
        
        values_away = [
            normalize(away_stats['off_rating'], 105, 125),
            normalize(130 - away_stats['def_rating'], 105, 125),
            normalize(away_stats['net_rating'] + 20, -10, 30),
            normalize(away_stats['effective_fg_pct'] * 100, 45, 60),
            normalize(away_stats['pace'], 95, 105),
            normalize(away_stats.get('consistency', 0.8) * 100, 50, 100)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_home,
            theta=categories,
            fill='toself',
            name=analysis['home_team'],
            line_color='#17408B',
            opacity=0.8
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values_away,
            theta=categories,
            fill='toself',
            name=analysis['away_team'],
            line_color='#C9082A',
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title='Comparação de Perfil das Equipes',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_analysis_tab(self, analysis: Dict):
        """Renderiza aba de análise detalhada"""
        
        # Informações do confronto
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Time da Casa")
            st.write(f"**Time:** {analysis['home_team']}")
            st.write(f"**Conferência:** {analysis['home_conference']}")
            st.write(f"**Recorde Casa:** {analysis['home_stats'].get('home_record', 'N/A')}")
            st.write(f"**Recorde Geral:** {analysis['home_stats']['win_pct']:.3f}")
        
        with col2:
            st.subheader("✈️ Time Visitante")
            st.write(f"**Time:** {analysis['away_team']}")
            st.write(f"**Conferência:** {analysis['away_conference']}")
            st.write(f"**Recorde Fora:** {analysis['away_stats'].get('away_record', 'N/A')}")
            st.write(f"**Recorde Geral:** {analysis['away_stats']['win_pct']:.3f}")
        
        st.markdown("---")
        
        # Análise dos 4 Fatores
        st.subheader("📊 ANÁLISE DOS 4 FATORES DE DEAN OLIVER")
        
        four_factors = analysis['four_factors_analysis']
        
        for factor_name, factor_data in four_factors.items():
            if factor_name == 'summary':
                continue
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{factor_name.upper()}**")
            
            with col2:
                home_val = factor_data['home']
                away_val = factor_data['away']
                
                if isinstance(home_val, float):
                    st.metric(
                        analysis['home_team'][:10], 
                        f"{home_val:.3f}",
                        f"{home_val - away_val:+.3f}" if home_val != away_val else ""
                    )
                else:
                    st.text(f"{analysis['home_team'][:10]}: {home_val}")
            
            with col3:
                if isinstance(away_val, float):
                    st.metric(
                        analysis['away_team'][:10], 
                        f"{away_val:.3f}",
                        f"{away_val - home_val:+.3f}" if home_val != away_val else ""
                    )
                else:
                    st.text(f"{analysis['away_team'][:10]}: {away_val}")
            
            # Barra de vantagem
            advantage = factor_data['advantage']
            if advantage != 'neutral':
                team_name = analysis['home_team'] if advantage == 'home' else analysis['away_team']
                impact = factor_data['impact']
                
                st.progress(
                    home_val if advantage == 'home' else away_val,
                    text=f"Vantagem para {team_name} ({impact})"
                )
            
            st.markdown("---")
        
        # Resumo dos 4 Fatores
        summary = four_factors['summary']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vantagens Casa", summary['home_advantages'])
        with col2:
            st.metric("Vantagens Visitante", summary['away_advantages'])
        with col3:
            dominant = summary['dominant_team']
            if dominant == 'home':
                st.success("🏠 DOMINÂNCIA DA CASA")
            elif dominant == 'away':
                st.error("✈️ DOMINÂNCIA DO VISITANTE")
            else:
                st.info("⚖️ EQUILÍBRIO")
        
        # Análise de Matchups
        st.markdown("---")
        st.subheader("🤼 ANÁLISE DE MATCHUPS")
        
        matchup_analysis = analysis['matchup_analysis']
        
        for matchup_name, matchup_data in matchup_analysis.items():
            with st.expander(f"{matchup_name.replace('_', ' ').title()}"):
                st.write(f"**Descrição:** {matchup_data['description']}")
                st.write(f"**Vantagem:** {matchup_data['advantage'].upper()}")
                st.write(f"**Magnitude:** {matchup_data.get('magnitude', 'N/A')}")
                
                if 'game_type' in matchup_data:
                    st.write(f"**Tipo de Jogo:** {matchup_data['game_type']}")
    
    def _render_insights_tab(self, analysis: Dict):
        """Renderiza aba de insights"""
        
        st.subheader("💡 INSIGHTS ESTRATÉGICOS")
        
        # Insights principais
        for i, insight in enumerate(analysis['insights'], 1):
            if i <= 3:
                st.success(f"**{i}.** {insight}")
            elif i <= 6:
                st.info(f"**{i}.** {insight}")
            else:
                st.warning(f"**{i}.** {insight}")
        
        # Recomendações de apostas
        st.markdown("---")
        st.subheader("💰 RECOMENDAÇÕES DE APOSTAS")
        
        prediction = analysis['final_prediction']
        win_prob = prediction['win_probability']
        margin = prediction['predicted_margin']
        total = prediction['predicted_total']
        confidence = prediction['confidence']
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            # Moneyline
            if win_prob > 0.65:
                st.success("**MONEYLINE:** 🔥 FORTE APOSTA")
                st.write(f"Probabilidade: {win_prob:.1%}")
                st.write(f"Confiança: {confidence:.0%}")
            elif win_prob > 0.55:
                st.warning("**MONEYLINE:** ⚠️ APOSTA MODERADA")
                st.write(f"Probabilidade: {win_prob:.1%}")
                st.write(f"Confiança: {confidence:.0%}")
            else:
                st.error("**MONEYLINE:** 🚫 EVITE APOSTAR")
                st.write("Jogo muito equilibrado")
        
        with rec_col2:
            # Spread
            abs_margin = abs(margin)
            if abs_margin > 8:
                if margin > 0:
                    st.success(f"**SPREAD:** CASA -{abs_margin:.1f}")
                else:
                    st.success(f"**SPREAD:** VISITANTE +{abs_margin:.1f}")
                st.write("Margem significativa")
            elif abs_margin > 4:
                st.warning(f"**SPREAD:** AJUSTE FINO")
                st.write(f"Margem: {margin:+.1f}")
            else:
                st.error("**SPREAD:** RISCO ALTO")
                st.write("Margem muito pequena")
        
        with rec_col3:
            # Over/Under
            if total > 225:
                st.success("**TOTAL:** 🔥 OVER")
                st.write(f"Projeção: {total:.1f}")
                st.write("Jogo ofensivo esperado")
            elif total > 215:
                st.warning("**TOTAL:** ⚠️ OVER LEVE")
                st.write(f"Projeção: {total:.1f}")
            elif total < 205:
                st.success("**TOTAL:** 🔥 UNDER")
                st.write(f"Projeção: {total:.1f}")
                st.write("Jogo defensivo esperado")
            else:
                st.warning("**TOTAL:** ⚠️ UNDER LEVE")
                st.write(f"Projeção: {total:.1f}")
        
        # Relatório final
        st.markdown("---")
        st.subheader("📄 RELATÓRIO FINAL")
        
        with st.expander("📋 VER RELATÓRIO COMPLETO"):
            st.write(f"**Análise:** {analysis['matchup']}")
            st.write(f"**Data:** {analysis['date']}")
            st.write(f"**Temporada:** {analysis['season']}")
            st.write(f"**ID da Análise:** {analysis['analysis_id']}")
            st.write("---")
            
            st.write("**CONFIGURAÇÃO DO JOGO:**")
            st.write(f"- Casa: {analysis['home_team']} ({analysis['home_conference']})")
            st.write(f"- Visitante: {analysis['away_team']} ({analysis['away_conference']})")
            st.write(f"- Tipo: {'Intra-Conferência' if analysis['home_conference'] == analysis['away_conference'] else 'Inter-Conferência'}")
            st.write(f"- Local: {analysis['venue']}")
            
            st.write("**RESULTADOS PRINCIPAIS:**")
            st.write(f"- Probabilidade de Vitória: {prediction['win_probability']:.1%}")
            st.write(f"- Margem Esperada: {prediction['predicted_margin']:+.1f} pontos")
            st.write(f"- Total de Pontos: {prediction['predicted_total']:.1f}")
            st.write(f"- Confiança do Modelo: {prediction['confidence']:.0%}")
            
            st.write("**FATORES DECISIVOS:**")
            for factor in analysis['four_factors_analysis'].values():
                if isinstance(factor, dict) and 'advantage' in factor:
                    st.write(f"- {factor.get('name', 'Fator')}: Vantagem para {factor['advantage']}")
            
            # Botão para exportar
            if st.button("📥 EXPORTAR RELATÓRIO (JSON)", key="export_json"):
                analysis_json = json.dumps(analysis, indent=2, default=str)
                st.download_button(
                    label="⬇️ BAIXAR RELATÓRIO",
                    data=analysis_json,
                    file_name=f"nba_analysis_{analysis['analysis_id']}.json",
                    mime="application/json",
                    key="download_json"
                )
    
    def render_homepage(self):
        """Renderiza página inicial"""
        
        st.markdown('<h1 class="main-header">NBA ULTIMATE ANALYTICS 2025-26</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <h3 style="color: #666;">SISTEMA PROFISSIONAL DE ANÁLISE ESTATÍSTICA</h3>
            <p style="color: #888; font-size: 1.1rem;">Temporada 2025-26 • Dados em Tempo Real • Machine Learning</p>
            <p style="color: #28a745; font-weight: bold;">✅ SELEÇÃO LIVRE DE TIMES - QUALQUER TIME PODE SER CASA OU VISITANTE</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cards de destaque
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🤖 IA AVANÇADA</h4>
                <p>Machine Learning com Random Forest para previsões precisas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🎲 10K SIMULAÇÕES</h4>
                <p>Análise Monte Carlo com 10,000 iterações por jogo</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 30+ MÉTRICAS</h4>
                <p>Análise completa com estatísticas avançadas da NBA</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Como usar
        st.subheader("🎯 COMO USAR O SISTEMA")
        
        steps_col1, steps_col2 = st.columns(2)
        
        with steps_col1:
            st.markdown("""
            ### 1️⃣ SELECIONE OS TIMES
            - **QUALQUER TIME** pode ser casa ou visitante
            - Sem restrições de conferência
            - Validação automática para evitar times iguais
            
            ### 2️⃣ CONFIGURE A ANÁLISE
            - Defina local do jogo (Casa/Visitante/Neutro)
            - Ajuste dias de descanso
            - Adicione fatores especiais
            """)
        
        with steps_col2:
            st.markdown("""
            ### 3️⃣ EXECUTE A ANÁLISE
            - Clique em "Executar Análise"
            - Sistema processa múltiplos modelos
            - Resultados em segundos
            
            ### 4️⃣ ANALISE OS RESULTADOS
            - Dashboard interativo
            - Gráficos profissionais
            - Insights estratégicos
            """)
        
        # Times em destaque
        st.markdown("---")
        st.subheader("🏀 TIMES EM DESTAQUE 2025-26")
        
        # Criar ranking simulado
        teams = self.data_engine.get_all_teams()
        top_teams = ["Boston Celtics", "Oklahoma City Thunder", "Denver Nuggets", 
                    "Milwaukee Bucks", "Minnesota Timberwolves"]
        
        for i, team in enumerate(top_teams[:5], 1):
            stats = self.data_engine.get_team_stats(team)
            conference = self.data_engine.get_conference(team)
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
            
            with col1:
                st.markdown(f"**#{i}**")
            with col2:
                st.markdown(f"**{team}**")
            with col3:
                st.markdown(f"**{conference}**")
            with col4:
                st.markdown(f"Win %: **{stats['win_pct']:.3f}**")
            with col5:
                st.markdown(f"Net Rtg: **{stats['net_rating']:+.1f}**")
        
        # Botão para começar
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h3>PRONTO PARA COMEÇAR?</h3>
            <p>Configure sua análise na barra lateral à esquerda</p>
            <p><strong>Lembrete:</strong> Agora você pode escolher QUALQUER time para QUALQUER posição!</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal do sistema"""
    
    # Inicializar UI
    nba_ui = NBAUltimateUI()
    
    # Renderizar sidebar
    nba_ui.render_sidebar()
    
    # Verificar se há análise atual
    current_analysis = st.session_state.get('current_analysis')
    
    if current_analysis:
        # Renderizar dashboard
        nba_ui.render_dashboard(current_analysis)
    else:
        # Renderizar homepage
        nba_ui.render_homepage()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem; padding: 20px;">
        <p><strong>NBA ULTIMATE ANALYTICS SYSTEM v3.1</strong> • Temporada 2025-26</p>
        <p><strong>✅ CORREÇÃO APLICADA:</strong> Seleção Livre de Times (Qualquer time pode ser casa ou visitante)</p>
        <p>Sistema Profissional de Análise Estatística • Desenvolvido para Alto Valor</p>
        <p>⚠️ Para fins educacionais e analíticos • Use com responsabilidade</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
