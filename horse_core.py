#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
马匹分析核心引擎 v3.5 - 实战增强版
新增：ML概率融合、Harville精确计算、专业回测、风控规则
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from itertools import combinations, permutations
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HorseCore:
    def __init__(self, data_path: str = "data/sample_races.csv"):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load_or_create()
        self._precompute()
        self.ml_model = None
        self.scaler = None
        self.feature_cols = ["draw", "weight", "distance", "class", "days_since",
                             "jockey_wr", "trainer_wr", "log_odds"]
        self.bankroll_history: List[Dict] = []
        # 风控默认参数
        self.max_stake_pct = 0.05          # 单注最大 5%
        self.daily_loss_limit_pct = 0.08   # 单日最大亏损 8%
        self.kelly_fraction = 0.25         # 1/4 凯利

    # ==================== 数据 ====================
    def _load_or_create(self) -> pd.DataFrame:
        if self.data_path.exists():
            df = pd.read_csv(self.data_path, parse_dates=["date"])
            return df

        rng = np.random.default_rng(42)
        horses = ["飞马神驹", "闪电侠", "金满堂", "威风凛凛", "幸运星",
                  "王者归来", "疾风", "天行者", "神勇骑", "银河战舰", "赤兔", "乌骓"]
        jockeys = ["潘顿", "莫雷拉", "何泽伦", "梁家俊", "田丰", "杨明纶", "巴度"]
        trainers = ["约翰摩亚", "蔡约翰", "罗富全", "方嘉柏", "大卫希斯", "吕健威"]
        venues = ["沙田", "跑马地"]
        goings = ["好地", "好地至快地", "好至软地"]
        distances = [1000, 1200, 1400, 1600, 1650, 1800]

        rows = []
        base = datetime(2026, 5, 1)
        for i in range(280):
            date = base + timedelta(days=int(rng.integers(0, 170)))
            horse = rng.choice(horses)
            venue = rng.choice(venues)
            dist = int(rng.choice(distances))
            place = int(rng.choice([1,2,3,4,5,6,7,8,9,10],
                                   p=[0.13,0.13,0.12,0.11,0.10,0.10,0.09,0.08,0.07,0.07]))
            draw = int(rng.integers(1, 15))
            weight = round(float(rng.uniform(117, 133)), 1)
            jockey = rng.choice(jockeys)
            trainer = rng.choice(trainers)
            base_t = {1000:56.7, 1200:69.7, 1400:82.1, 1600:94.9, 1650:99.4, 1800:107.8}[dist]
            time_sec = round(base_t + (place-1)*0.32 + float(rng.normal(0, 0.35)), 2)
            odds = round(max(1.6, 3.2 + (place-1)*1.7 + float(rng.normal(0, 1.3))), 1)
            going = rng.choice(goings)
            cls = int(rng.integers(1, 5))
            rows.append([horse, date, venue, dist, place, draw, weight,
                         jockey, trainer, time_sec, odds, going, cls])

        cols = ["horse","date","venue","distance","place","draw","weight",
                "jockey","trainer","time_sec","odds","going","class"]
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(self.data_path, index=False)
        return df

    def load_external_csv(self, path: str):
        df = pd.read_csv(path, parse_dates=["date"])
        required = {"horse","date","venue","distance","place","draw",
                    "weight","jockey","trainer","time_sec","odds"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")
        for col in ["going", "class"]:
            if col not in df.columns:
                df[col] = "好地" if col == "going" else 3
        self.df = df
        self._precompute()
        self.ml_model = None  # 需要重新训练

    def _precompute(self):
        self.jockey_stats = self.df.groupby("jockey").agg(
            runs=("place","count"), wins=("place", lambda x: (x==1).sum())
        )
        self.jockey_stats["win_rate"] = self.jockey_stats["wins"] / self.jockey_stats["runs"]

        self.trainer_stats = self.df.groupby("trainer").agg(
            runs=("place","count"), wins=("place", lambda x: (x==1).sum())
        )
        self.trainer_stats["win_rate"] = self.trainer_stats["wins"] / self.trainer_stats["runs"]

        self.combo_stats = self.df.groupby(["jockey","trainer"]).agg(
            runs=("place","count"), wins=("place", lambda x: (x==1).sum())
        )
        self.combo_stats["win_rate"] = self.combo_stats["wins"] / self.combo_stats["runs"].clip(1)

        self.draw_bias = (
            self.df.groupby(["venue","distance","draw"])
            .agg(runs=("place","count"),
                 wins=("place", lambda x: (x==1).sum()),
                 top3=("place", lambda x: (x<=3).sum()))
            .reset_index()
        )
        self.draw_bias["win_rate"] = self.draw_bias["wins"] / self.draw_bias["runs"]
        self.draw_bias["top3_rate"] = self.draw_bias["top3"] / self.draw_bias["runs"]

    # ==================== 评分 ====================
    def form_score(self, horse: str, n: int = 5) -> float:
        recent = self.df[self.df["horse"]==horse].sort_values("date", ascending=False).head(n)
        if recent.empty:
            return 42.0
        scores, weights = [], []
        for i, row in enumerate(recent.itertuples()):
            s = {1:100,2:78,3:58,4:38,5:22}.get(int(row.place), 10)
            days = (datetime.now() - pd.Timestamp(row.date)).days
            if days > 60: s *= 0.82
            elif days > 45: s *= 0.92
            scores.append(s)
            weights.append(1.0/(i+1.15))
        return round(float(np.average(scores, weights=weights)), 1)

    def speed_figure(self, horse: str, target_dist: int = 1200) -> float:
        hdf = self.df[self.df["horse"]==horse]
        if hdf.empty: return 50.0
        figs = []
        for _, r in hdf.iterrows():
            adj = r["time_sec"] * (target_dist / r["distance"])
            adj -= (126 - r["weight"]) * 0.02
            figs.append(80 + (70.0 - adj) * 12)
        return round(float(np.mean(figs[-6:])), 1)

    def distance_suit(self, horse: str, dist: int) -> float:
        hdf = self.df[self.df["horse"]==horse]
        close = hdf[abs(hdf["distance"]-dist) <= 200]
        if close.empty: return 48.0
        wr = (close["place"]==1).mean()
        t3 = (close["place"]<=3).mean()
        return round(40 + wr*40 + t3*20, 1)

    def going_suit(self, horse: str, going: str = "好地") -> float:
        hdf = self.df[self.df["horse"]==horse]
        same = hdf[hdf["going"].str.contains(str(going)[:2], na=False)]
        if len(same) < 2: return 50.0
        return round(40 + (same["place"]==1).mean()*40 + (same["place"]<=3).mean()*20, 1)

    def connection_score(self, jockey: str, trainer: str) -> float:
        j_wr = self.jockey_stats.loc[jockey,"win_rate"] if jockey in self.jockey_stats.index else 0.11
        t_wr = self.trainer_stats.loc[trainer,"win_rate"] if trainer in self.trainer_stats.index else 0.11
        key = (jockey, trainer)
        c_wr = self.combo_stats.loc[key,"win_rate"] if key in self.combo_stats.index else 0.11
        return round(min((j_wr*0.35 + t_wr*0.35 + c_wr*0.30)*180, 92), 1)

    def draw_score(self, venue: str, distance: int, draw: int) -> float:
        sub = self.draw_bias[
            (self.draw_bias["venue"]==venue) &
            (self.draw_bias["distance"]==distance) &
            (self.draw_bias["draw"]==draw)
        ]
        if sub.empty or sub.iloc[0]["runs"] < 3:
            if venue == "沙田" and distance <= 1200:
                return 55 + max(0, 8 - draw) * 2.2
            if venue == "跑马地" and distance <= 1200:
                return 52 + max(0, 6 - draw) * 1.8
            return 50.0
        return round(40 + sub.iloc[0]["win_rate"]*120, 1)

    def composite(self, horse, jockey, trainer, distance=1200, going="好地",
                  venue="沙田", draw=5) -> Dict:
        form = self.form_score(horse)
        speed = self.speed_figure(horse, distance)
        dist = self.distance_suit(horse, distance)
        go = self.going_suit(horse, going)
        conn = self.connection_score(jockey, trainer)
        dr = self.draw_score(venue, distance, draw)
        total = form*0.25 + speed*0.22 + dist*0.14 + go*0.10 + conn*0.16 + dr*0.13
        return {"form":form, "speed":speed, "distance":dist, "going":go,
                "connection":conn, "draw":dr, "total":round(total,1)}

    # ==================== 概率 ====================
    def softmax(self, scores: List[float], temp: float = 11.0) -> np.ndarray:
        s = np.array(scores, dtype=float) / temp
        e = np.exp(s - np.max(s))
        return e / e.sum()

    def kelly(self, prob: float, odds: float) -> float:
        if odds <= 1 or prob <= 0: return 0.0
        b = odds - 1
        k = (b * prob - (1 - prob)) / b
        return max(0.0, k * self.kelly_fraction)

    def apply_risk_control(self, kelly_pct: float, bankroll: float) -> float:
        """风控：单注不超过 max_stake_pct"""
        max_pct = self.max_stake_pct * 100
        return min(kelly_pct, max_pct)

    # ==================== Harville 公式 ====================
    def harville_place(self, win_probs: np.ndarray) -> np.ndarray:
        """位置（前3）概率近似"""
        n = len(win_probs)
        place = np.zeros(n)
        for i in range(n):
            p_i = win_probs[i]
            # 第一名 + 第二名 + 第三名
            p2 = 0.0
            for j in range(n):
                if j == i: continue
                p2 += win_probs[j] * p_i / (1 - win_probs[j] + 1e-9)
            p3 = 0.0
            for j in range(n):
                if j == i: continue
                for k in range(n):
                    if k == i or k == j: continue
                    p3 += (win_probs[j] * win_probs[k] * p_i) / (
                        (1 - win_probs[j] + 1e-9) * (1 - win_probs[j] - win_probs[k] + 1e-9))
            place[i] = min(0.92, p_i + p2/2 + p3/3)  # 简化归一
        place = place / place.sum() * min(2.8, n * 0.28)
        return np.clip(place, 0.04, 0.78)

    def harville_exacta(self, win_probs: np.ndarray, i: int, j: int) -> float:
        """连赢（无序）= P(i1st j2nd) + P(j1st i2nd)"""
        if i == j: return 0.0
        p_ij = win_probs[i] * win_probs[j] / (1 - win_probs[i] + 1e-9)
        p_ji = win_probs[j] * win_probs[i] / (1 - win_probs[j] + 1e-9)
        return p_ij + p_ji

    def harville_trifecta(self, win_probs: np.ndarray, i: int, j: int, k: int) -> float:
        if len({i,j,k}) < 3: return 0.0
        p = (win_probs[i] * win_probs[j] / (1 - win_probs[i] + 1e-9) *
             win_probs[k] / (1 - win_probs[i] - win_probs[j] + 1e-9))
        return p

    # ==================== 整场分析（融合 ML） ====================
    def analyze_race(self, runners: List[Dict], distance: int = 1200,
                     going: str = "好地", venue: str = "沙田",
                     bankroll: float = 10000, ml_weight: float = 0.35) -> pd.DataFrame:
        rows = []
        for r in runners:
            sc = self.composite(
                r["horse"], r.get("jockey","未知"), r.get("trainer","未知"),
                distance, going, venue, r.get("draw", 5)
            )
            rows.append({**r, **sc})

        df = pd.DataFrame(rows)
        if df.empty: return df

        score_probs = self.softmax(df["total"].tolist())

        # ML 融合
        if self.ml_model is not None:
            ml_probs = []
            for _, row in df.iterrows():
                ml_probs.append(self._ml_predict_row(row, distance))
            ml_probs = np.array(ml_probs)
            ml_probs = ml_probs / ml_probs.sum()  # 归一
            final_probs = (1 - ml_weight) * score_probs + ml_weight * ml_probs
            final_probs = final_probs / final_probs.sum()
        else:
            final_probs = score_probs

        df["model_prob"] = np.round(final_probs * 100, 1)
        df["implied"] = np.round(100 / df["odds"], 1)
        df["edge"] = np.round(df["model_prob"] - df["implied"], 1)
        df["ev"] = np.round((df["model_prob"]/100 * df["odds"] - 1) * 100, 1)

        kelly_raw = [self.kelly(p/100, o)*100 for p,o in zip(df["model_prob"], df["odds"])]
        df["kelly_pct"] = [round(self.apply_risk_control(k, bankroll), 2) for k in kelly_raw]
        df["stake"] = (df["kelly_pct"]/100 * bankroll).round(0).astype(int)

        # 位置概率
        place_p = self.harville_place(final_probs)
        df["place_prob"] = np.round(place_p * 100, 1)

        df = df.sort_values("edge", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df)+1)

        def rec(row):
            if row["edge"] > 8 and row["ev"] > 12: return "🔥 强烈推荐"
            if row["edge"] > 4 and row["ev"] > 6:  return "✅ 推荐"
            if row["edge"] > 1.5: return "👀 可关注"
            return "❌ 价值不足"
        df["advice"] = df.apply(rec, axis=1)
        return df

    def quinella_suggestions(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        probs = (df["model_prob"] / 100).values
        names = df["horse"].values
        combos = []
        for i, j in combinations(range(min(top_n, len(df))), 2):
            p = self.harville_exacta(probs, i, j)
            combos.append({
                "连赢组合": f"{names[i]} + {names[j]}",
                "估算概率%": round(p*100, 2),
                "马1": names[i], "马2": names[j]
            })
        return pd.DataFrame(sorted(combos, key=lambda x: -x["估算概率%"])).head(8)

    def trifecta_suggestions(self, df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
        probs = (df["model_prob"] / 100).values
        names = df["horse"].values
        results = []
        for perm in permutations(range(min(top_n, len(df))), 3):
            p = self.harville_trifecta(probs, *perm)
            results.append({
                "三重彩": f"{names[perm[0]]} → {names[perm[1]]} → {names[perm[2]]}",
                "估算概率%": round(p*100, 3)
            })
        return pd.DataFrame(sorted(results, key=lambda x: -x["估算概率%"])).head(10)

    # ==================== 机器学习 ====================
    def train_ml(self):
        df = self.df.copy()
        df["won"] = (df["place"] == 1).astype(int)
        max_date = df["date"].max()
        df["days_since"] = (max_date - df["date"]).dt.days
        df["jockey_wr"] = df["jockey"].map(self.jockey_stats["win_rate"]).fillna(0.12)
        df["trainer_wr"] = df["trainer"].map(self.trainer_stats["win_rate"]).fillna(0.12)
        df["log_odds"] = np.log(df["odds"].clip(1.1))

        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df["won"]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if HAS_LGB:
            self.ml_model = lgb.LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42
            )
        else:
            self.ml_model = LogisticRegression(max_iter=600, C=0.8)
        self.ml_model.fit(X_scaled, y)
        return "LightGBM" if HAS_LGB else "LogisticRegression"

    def _ml_predict_row(self, row: pd.Series, distance: int) -> float:
        if self.ml_model is None or self.scaler is None:
            return 0.12
        j_wr = self.jockey_stats.loc[row["jockey"], "win_rate"] if row["jockey"] in self.jockey_stats.index else 0.12
        t_wr = self.trainer_stats.loc[row["trainer"], "win_rate"] if row["trainer"] in self.trainer_stats.index else 0.12
        feats = np.array([[
            row.get("draw", 6),
            row.get("weight", 125),
            distance,
            3,
            18,
            j_wr, t_wr,
            np.log(max(row.get("odds", 5), 1.1))
        ]])
        feats = self.scaler.transform(feats)
        return float(self.ml_model.predict_proba(feats)[0, 1])

    # ==================== 专业回测 ====================
    def walk_forward_backtest(self, min_edge: float = 4.0,
                              init_bankroll: float = 10000,
                              stake_mode: str = "kelly") -> Dict:
        """严格按时间顺序的 walk-forward 回测"""
        df = self.df.sort_values("date").reset_index(drop=True)
        dates = sorted(df["date"].unique())
        if len(dates) < 40:
            return {"error": "数据量不足，至少需要约40个赛日"}

        bankroll = init_bankroll
        equity = [bankroll]
        bets = []
        daily_pnl = {}

        for i, d in enumerate(dates[25:]):  # 留足历史
            hist = df[df["date"] < d]
            race = df[df["date"] == d]
            if len(race) < 5:
                continue

            # 用历史重算简单统计（避免未来函数）
            j_stats = hist.groupby("jockey")["place"].apply(lambda x: (x==1).mean())
            t_stats = hist.groupby("trainer")["place"].apply(lambda x: (x==1).mean())

            for _, r in race.iterrows():
                h_hist = hist[hist["horse"] == r["horse"]]
                if len(h_hist) < 2:
                    continue
                est_p = (h_hist["place"] == 1).mean()
                # 轻微 Shrinkage
                est_p = 0.7 * est_p + 0.3 * 0.12
                implied = 1.0 / r["odds"]
                edge = (est_p - implied) * 100

                if edge >= min_edge:
                    if stake_mode == "kelly":
                        k = self.kelly(est_p, r["odds"])
                        stake = min(bankroll * k, bankroll * self.max_stake_pct)
                    else:
                        stake = min(200, bankroll * 0.02)

                    if stake < 10:
                        continue

                    won = r["place"] == 1
                    profit = stake * (r["odds"] - 1) if won else -stake
                    bankroll += profit
                    equity.append(bankroll)
                    bets.append({
                        "date": d, "horse": r["horse"], "odds": r["odds"],
                        "edge": round(edge,1), "stake": round(stake,1),
                        "profit": round(profit,1), "won": won
                    })
                    day_key = str(d.date()) if hasattr(d, "date") else str(d)[:10]
                    daily_pnl[day_key] = daily_pnl.get(day_key, 0) + profit

        if not bets:
            return {"bets": 0, "final_bankroll": init_bankroll, "roi": 0}

        bet_df = pd.DataFrame(bets)
        total_stake = bet_df["stake"].sum()
        net = bet_df["profit"].sum()
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        max_dd = ((equity - peak) / peak * 100).min()

        return {
            "bets": len(bet_df),
            "win_rate": round(bet_df["won"].mean()*100, 1),
            "net_profit": round(net, 1),
            "roi": round(net / total_stake * 100, 1),
            "final_bankroll": round(bankroll, 1),
            "max_drawdown_pct": round(max_dd, 1),
            "equity_curve": equity.tolist(),
            "bet_log": bet_df
        }

    # ==================== 其他工具 ====================
    def draw_bias_report(self, venue: str, distance: int) -> pd.DataFrame:
        sub = self.draw_bias[
            (self.draw_bias["venue"]==venue) & (self.draw_bias["distance"]==distance)
        ].copy()
        if sub.empty: return pd.DataFrame()
        sub = sub.sort_values("draw")
        sub["建议"] = sub["win_rate"].apply(
            lambda x: "优势档" if x > 0.16 else ("中性" if x > 0.09 else "劣势档")
        )
        return sub[["draw","runs","wins","win_rate","top3_rate","建议"]]

    def update_bankroll(self, stake: float, profit: float):
        self.bankroll_history.append({
            "time": datetime.now(), "stake": stake, "profit": profit
        })

    def bankroll_summary(self, current: float) -> Dict:
        if not self.bankroll_history:
            return {"total_bets":0, "net_profit":0, "roi":0, "current":current}
        hist = pd.DataFrame(self.bankroll_history)
        net = hist["profit"].sum()
        total_st = hist["stake"].sum()
        return {
            "total_bets": len(hist),
            "net_profit": round(net,1),
            "roi": round(net/total_st*100,1) if total_st else 0,
            "current": current
        }
