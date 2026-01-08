import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ==========================================
# 設定 & 定数
# ==========================================
st.set_page_config(page_title="Chimera Phoenix Dashboard", layout="wide")
st.title("🦅 Chimera Phoenix Mk.V Dashboard")

TICKERS = ['^NDX', 'VUSTX', 'GC=F', '^GSPC', '^IRX']
START_DATE = '1986-01-01'
LEVERAGE_COST = 0.025

# パラメータ
SHIELD_EMA_SHORT = 20
SHIELD_EMA_LONG = 30
TREND_SMA_WINDOW = 200
MINVAR_LOOKBACK = 60
BULL_ALLOCATION = 0.60
BEAR_MAX_TQQQ = 0.20

# ==========================================
# 関数定義
# ==========================================
@st.cache_data(ttl=3600*12) # 12時間キャッシュ
def get_data():
    df_raw = yf.download(TICKERS, start=START_DATE, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        try: df = df_raw['Adj Close'].copy()
        except KeyError: df = df_raw['Close'].copy()
    else:
        df = df_raw.copy()
    return df.ffill()

def calculate_strategy(df):
    # 擬似データ生成
    daily_ret = df.pct_change().fillna(0)
    daily_drag = LEVERAGE_COST / 252
    
    syn_ret = pd.DataFrame(index=df.index)
    syn_ret['TQQQ'] = (daily_ret['^NDX'] * 3) - daily_drag
    syn_ret['TMF'] = (daily_ret['VUSTX'] * 3) - daily_drag
    syn_ret['TMV'] = (daily_ret['VUSTX'] * -3) - daily_drag
    syn_ret['GLD'] = daily_ret['GC=F']
    cash_yield = df['^IRX'].fillna(0) / 100
    syn_ret['CASH'] = cash_yield / 252

    # 価格データ構築
    ndx_price = df['^NDX'].fillna(method='ffill')
    syn_price = pd.DataFrame(index=df.index)
    syn_price['TMV'] = (1 + syn_ret['TMV']).cumprod() * 100
    
    # A. 債券盾 (Bond Shield)
    tmv_p = syn_price['TMV']
    ema_s = tmv_p.ewm(span=SHIELD_EMA_SHORT, adjust=False).mean()
    ema_l = tmv_p.ewm(span=SHIELD_EMA_LONG, adjust=False).mean()
    daily_bond_sig = (ema_s > ema_l).astype(int)
    monthly_bond_sig = daily_bond_sig.resample('M').last()
    eff_bond_sig = monthly_bond_sig.reindex(syn_ret.index).shift(1).ffill().fillna(0)
    bond_shield_ret = (syn_ret['TMV'] * eff_bond_sig) + (syn_ret['TMF'] * (1 - eff_bond_sig))

    # B. 黄金の盾 (Golden Shield)
    rolling_bond = (1 + bond_shield_ret).rolling(60).apply(np.prod, raw=True) - 1
    rolling_gold = (1 + syn_ret['GLD']).rolling(60).apply(np.prod, raw=True) - 1
    daily_gold_sig = (rolling_gold > rolling_bond).astype(int)
    monthly_gold_sig = daily_gold_sig.resample('M').last()
    eff_gold_sig = monthly_gold_sig.reindex(syn_ret.index).shift(1).ffill().fillna(0)
    final_shield_ret = (syn_ret['GLD'] * eff_gold_sig) + (bond_shield_ret * (1 - eff_gold_sig))
    syn_ret['Shield'] = final_shield_ret

    # C. トレンド判定
    ndx_sma = ndx_price.rolling(window=TREND_SMA_WINDOW).mean()
    is_bull_daily = ndx_price > ndx_sma
    is_bull_monthly = is_bull_daily.resample('M').last()

    # バックテスト計算
    history = []
    cash = 10000.0
    monthly_dates = syn_ret.resample('M').last().index
    
    # 最新シグナル用変数
    last_signal = {}

    for i, date in enumerate(monthly_dates):
        if syn_ret.index.get_loc(date, method='ffill') < 250: continue
        if date not in syn_ret.index: date = syn_ret.index[syn_ret.index.get_loc(date, method='ffill')]
        
        # 配分比率
        idx_end = syn_ret.index.get_loc(date)
        idx_start = idx_end - MINVAR_LOOKBACK
        sub_tqqq = syn_ret['TQQQ'].iloc[idx_start:idx_end]
        sub_shield = syn_ret['Shield'].iloc[idx_start:idx_end]
        
        vol_a = sub_tqqq.std()
        vol_b = sub_shield.std()
        corr = sub_tqqq.corr(sub_shield)
        
        try: minvar_w = (vol_b**2 - corr*vol_a*vol_b) / (vol_a**2 + vol_b**2 - 2*corr*vol_a*vol_b)
        except: minvar_w = 0.5
        minvar_w = max(0.0, min(1.0, minvar_w))
        
        bull_mode = is_bull_monthly.asof(date)
        
        if bull_mode:
            final_w_tqqq = BULL_ALLOCATION
            mode_str = "Bull"
        else:
            final_w_tqqq = min(minvar_w, BEAR_MAX_TQQQ)
            mode_str = "Bear"
            
        w_shield = 1.0 - final_w_tqqq

        # 翌月リターン
        if i < len(monthly_dates) - 1:
            next_date = monthly_dates[i+1]
            if next_date not in syn_ret.index:
                next_date = syn_ret.index[syn_ret.index.get_loc(next_date, method='ffill')]
            
            mask = (syn_ret.index > date) & (syn_ret.index <= next_date)
            r_tqqq = (1 + syn_ret.loc[mask, 'TQQQ']).prod() - 1
            r_shield = (1 + syn_ret.loc[mask, 'Shield']).prod() - 1
            port_ret = (r_tqqq * final_w_tqqq) + (r_shield * w_shield)
            cash *= (1 + port_ret)
            
            # 盾の中身
            is_gold = eff_gold_sig.loc[next_date] == 1
            is_tmv = eff_bond_sig.loc[next_date] == 1
            shield_name = "GLD" if is_gold else ("TMV" if is_tmv else "TMF")

            history.append({'Date': next_date, 'Asset': cash, 'Shield': shield_name, 'TQQQ_Ratio': final_w_tqqq})

            # 最新シグナル保存 (最後のループで上書きされる)
            last_signal = {
                'Trend': mode_str,
                'Shield': shield_name,
                'TQQQ_Alloc': final_w_tqqq,
                'Shield_Alloc': w_shield,
                'NDX_Price': ndx_price.loc[date],
                'NDX_SMA': ndx_sma.loc[date]
            }

    return pd.DataFrame(history).set_index('Date'), last_signal

# ==========================================
# メイン処理
# ==========================================
df = get_data()

if not df.empty:
    with st.spinner('Calculating Strategy...'):
        res_df, signal = calculate_strategy(df)

    # --- 1. 最新シグナル表示 ---
    st.header("📢 Current Signal (Next Month)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Market Trend", signal['Trend'], 
                  delta=f"NDX vs SMA: {(signal['NDX_Price']/signal['NDX_SMA']-1)*100:.1f}%")
    with col2:
        st.metric("TQQQ Allocation", f"{signal['TQQQ_Alloc']*100:.1f}%")
    with col3:
        st.metric("Current Shield", signal['Shield'], f"Alloc: {signal['Shield_Alloc']*100:.1f}%")

    # --- 2. パフォーマンスチャート ---
    st.header("📈 Long-Term Performance (Log Scale)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Asset'], mode='lines', name='Strategy', line=dict(color='gold', width=2)))
    fig.update_layout(yaxis_type="log", height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. 統計データ ---
    years = (res_df.index[-1] - res_df.index[0]).days / 365.25
    final_val = res_df['Asset'].iloc[-1]
    cagr = (final_val / 10000.0) ** (1/years) - 1
    max_dd = ((res_df['Asset'].cummax() - res_df['Asset']) / res_df['Asset'].cummax()).max()
    
    st.subheader("Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Final Asset (from $10k)", f"${final_val:,.0f}")
    m2.metric("CAGR (Yearly)", f"{cagr*100:.2f}%")
    m3.metric("Max Drawdown", f"{max_dd*100:.2f}%")

    # --- 4. 月次リターン表 ---
    st.header("📅 Monthly Returns")
    
    monthly_ret = res_df['Asset'].pct_change()
    monthly_table = pd.DataFrame()
    monthly_table['Year'] = monthly_ret.index.year
    monthly_table['Month'] = monthly_ret.index.month
    monthly_table['Return'] = monthly_ret

    pivot_table = monthly_table.pivot_table(index='Year', columns='Month', values='Return')
    pivot_table = pivot_table * 100 # %表記に
    
    # 色付け関数
    def color_surv(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'

    st.dataframe(pivot_table.style.format("{:.1f}%").applymap(color_surv), height=600)

else:
    st.error("Data fetch failed. Please try again later.")
