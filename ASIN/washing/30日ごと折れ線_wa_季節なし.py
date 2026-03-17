import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt

# ==========================================
# 0. 表示・フォントの設定
# ==========================================
plt.rcParams["font.family"] = "MS Gothic"

# 以前のご指定に合わせた大きなフォント設定
font_size_global = 26
title_size = font_size_global
label_size = font_size_global
tick_size = font_size_global - 2
legend_size = font_size_global - 10

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\washing'
spec_file_name = 'ASIN+spec_wa.xlsx'
wa_daily_path = os.path.join(base_dir, 'wa_daily.csv')
wa_weekly_path = os.path.join(base_dir, 'wa_weekly.csv')
spec_file_path = os.path.join(base_dir, spec_file_name)

max_lag = 30
asins = list(dict.fromkeys(["B0BP6ZCQC5", "B09TSBGDN4", "B08Q7Q29CQ", "B08B1GLD8L", "B0B313BKHK",
           "B0B31621RB", "B084K7PM6J", "B07JR174ZT", "B09J8MF636", "B0BQQKTMVD",
           "B07MZLN15F", "B07RWR7BX6", "B084K7PM6J", "B07MZLN15F", "B07RWR7BX6", 
           "B08T1DM8KQ", "B0815PXPKZ", "B08KH6QFYZ", "B01C10VSM0", "B08B13BXQL", 
           "B094VFZXMG", "B01G58K3KM", "B0BP6ZS2TX", "B0BP6VPD4K", "B09N71PGTB", 
           "B09N73WQ2V", "B0BD3GJGLF", "B0BD3FYJ3S", "B0BH8JRJCQ", "B075B5WR85", 
           "B083N9LZ6Y", "B0BP6J3SQX", "B07LBWQ3BF", "B01C10VTE2", "B07YLRMKRG", 
           "B07JQX29XR", "B0BLMJJ9TV", "B09DKK6VCL", "B09P31T6GF", "B083PR3Y2K", 
           "B07KCNJY79", "B09DSQQGT8", "B07SQCZRCG", "B09DKKZHFC", "B07YC9Z3N1", 
           "B07YCBRJFQ", "B07YC9Z3MV", "B09NNH112Q", "B0956GJ3CL", "B0BPM6K84W", 
           "B0BPM8TQLZ", "B09571SNLK", "B0956PSK2H", "B0956MW64T", "B09G2L68H2", 
           "B09G2KS69R", "B09G2LJQ98", "B09G2LNCXG", "B09DKGXYCV", "B01HUJES06", 
           "B0162EV9QQ", "B07ZPSWGYN", "B07CDRZYM5"]))

lag_vars = [f'ln_y_lag{i}' for i in range(1, max_lag + 1)]
inter_vars = ['inter_cap', 'inter_aut', 'inter_pri', 'inter_siz']

# --- 変数リストの分離 ---
# ベースライン：ラグ変数のみ（季節性トレンド単体を除外）
# 提案モデル：ラグ変数 ＋ 相互作用項（この中に季節性指数が含まれる）
# ==========================================

# ==========================================
# 2. データ処理（トレンド構築・統合）
# ==========================================
wa_weekly = pd.read_csv(wa_weekly_path, encoding='cp932')
wa_weekly.columns = wa_weekly.columns.str.strip() 
wa_weekly['date'] = pd.to_datetime(wa_weekly['date'])
wa_weekly = wa_weekly.sort_values('date').set_index('date')

daily_range = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
wa_weekly_interp = wa_weekly.reindex(daily_range).interpolate(method='linear').reset_index().rename(columns={'index': 'date'})

wa_daily = pd.read_csv(wa_daily_path, encoding='cp932')
wa_daily.columns = wa_daily.columns.str.strip()
wa_daily['date'] = pd.to_datetime(wa_daily['date'])

search_adj = wa_daily.merge(wa_weekly_interp[['date', 'f_sum']], on='date', how='left')
num_cols_adj = search_adj.select_dtypes(include=[np.number]).columns
search_adj[num_cols_adj] = search_adj[num_cols_adj].interpolate(method='linear', limit_direction='both')

# 各特徴検索量に季節性(f_sum)を掛け合わせて調整済トレンドを作成
for fk in ['f_cap', 'f_aut', 'f_pri', 'f_siz']:
    search_adj[f'adj_{fk}'] = search_adj[fk] * search_adj['f_sum']

search_final = search_adj[['date', 'adj_f_cap', 'adj_f_aut', 'adj_f_pri', 'adj_f_siz', 'f_sum']]

specs_master = pd.read_excel(spec_file_path, usecols=[0, 1, 2, 3, 4])
specs_master.columns = ['asin', 'cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']

def get_numeric_scalar(val):
    num = pd.to_numeric(val, errors='coerce')
    return float(num) if (pd.notna(num) and num > 0) else np.nan

all_data = []
eps = 1e-6

for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_{asin}.csv')
    if not os.path.exists(file_path): continue
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(search_final, on='date', how='left')
    df = df.sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec_row = spec_match.iloc[0]
    
    k_c, A_c = 0.684, 11.74
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it'] + eps)

    new_cols = {}
    for i in range(1, max_lag + 1):
        new_cols[f'ln_y_lag{i}'] = df['ln_y'].shift(i)
    
    # モデルには入れないが計算には使用
    ln_z_cap = np.log(df['adj_f_cap'] + eps)
    ln_z_aut = np.log(df['adj_f_aut'] + eps)
    ln_z_pri = np.log(df['adj_f_pri'] + eps)
    ln_z_siz = np.log(df['adj_f_siz'] + eps)

    f_cap_val = get_numeric_scalar(spec_row['cap_spec'])
    f_pri_val = get_numeric_scalar(spec_row['pri_spec'])
    f_siz_val = get_numeric_scalar(spec_row['siz_spec'])
    d_aut_val = pd.to_numeric(spec_row['aut_spec'], errors='coerce')
    
    f_cap_s = np.log(f_cap_val) if pd.notna(f_cap_val) else 0.0
    f_pri_s = np.log(f_pri_val) if pd.notna(f_pri_val) else 0.0
    f_siz_s = np.log(f_siz_val) if pd.notna(f_siz_val) else 0.0
    d_aut_s = d_aut_val if pd.notna(d_aut_val) else 0.0

    # 3重相当の相互作用項（この中に季節性f_sumが内包されている）
    new_cols['inter_cap'] = ln_z_cap * f_cap_s
    new_cols['inter_aut'] = ln_z_aut * d_aut_s 
    new_cols['inter_pri'] = ln_z_pri * f_pri_s
    new_cols['inter_siz'] = ln_z_siz * f_siz_s

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    all_data.append(df)

full_df = pd.concat(all_data).dropna().sort_values('date')
asin_dummies = pd.get_dummies(full_df['asin'], drop_first=True).astype(float)
full_df = pd.concat([full_df, asin_dummies], axis=1)

# ==========================================
# 4. 学習と予測
# ==========================================
train_data = full_df[full_df['date'].dt.year == 2023].copy()
test_overall = full_df[full_df['date'].dt.year >= 2024].copy()
dummy_cols = asin_dummies.columns.tolist()

# 変数セットの定義
X_baseline_cols = lag_vars + dummy_cols
X_proposed_cols = lag_vars + inter_vars + dummy_cols

model_b = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_baseline_cols])).fit()
model_p = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_proposed_cols])).fit()

# 結果出力（ARは主要なものに限定）
print("\n" + "="*50)
print("提案モデル：主要変数の重み（回帰係数）")
print("="*50)
for lag in ['ln_y_lag1', 'ln_y_lag7', 'ln_y_lag30']:
    print(f"  {lag:15}: {model_p.params[lag]:8.4f}")

print(f"\n[外部需要・相互作用項]")
for var in inter_vars:
    coef = model_p.params[var]
    pval = model_p.pvalues[var]
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {var:15}: {coef:8.4f} (p={pval:.4f}) {sig}")

# 予測ループ
mae_results = []
current_date = test_overall['date'].min()
while current_date <= test_overall['date'].max():
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= current_date + pd.Timedelta(days=29))]
    if not batch.empty:
        # 学習時と同じカラム順序を保証
        exog_b = sm.add_constant(batch[X_baseline_cols], has_constant='add')[model_b.params.index]
        exog_p = sm.add_constant(batch[X_proposed_cols], has_constant='add')[model_p.params.index]
        
        pred_b = model_b.predict(exog_b)
        pred_p = model_p.predict(exog_p)
        
        mae_results.append({
            'テスト開始': current_date, 
            'MAE_ベースライン': mean_absolute_error(batch['ln_y'], pred_b),
            'MAE_提案モデル': mean_absolute_error(batch['ln_y'], pred_p)
        })
    current_date += pd.Timedelta(days=30)

err_df = pd.DataFrame(mae_results)
print("\n" + "="*50)
print(f"ベースライン平均 MAE: {err_df['MAE_ベースライン'].mean():.4f}")
print(f"提案モデル平均 MAE  : {err_df['MAE_提案モデル'].mean():.4f}")
print(f"改善率              : {(1 - err_df['MAE_提案モデル'].mean()/err_df['MAE_ベースライン'].mean())*100:.2f}%")
print("="*50)

# ==========================================
# 5. 可視化
# ==========================================
if not err_df.empty:
    fig, ax1 = plt.subplots(figsize=(16, 9), dpi=100)
    
    # 左軸：実績値
    daily_avg = full_df.groupby('date')['ln_y'].mean().reset_index()
    ax1.plot(daily_avg['date'], daily_avg['ln_y'], color='#FFCCCC', linewidth=1.5, label='相対売上(平均実績)', alpha=0.7)
    ax1.set_xlabel('日付', fontsize=label_size)
    ax1.set_ylabel('相対売上（ln_y）', fontsize=label_size)
    ax1.tick_params(axis='both', labelsize=tick_size)
    
    # 右軸：誤差（0-0.5に固定）
    ax2 = ax1.twinx()
    ax2.plot(err_df['テスト開始'], err_df['MAE_提案モデル'], color='#D62728', marker='o', markersize=10, label='MAE：提案モデル', linewidth=3)
    ax2.plot(err_df['テスト開始'], err_df['MAE_ベースライン'], color='#1F77B4', marker='s', markersize=10, label='MAE：ベースライン', linewidth=3)
    ax2.set_ylabel('予測誤差（MAE）', fontsize=label_size)
    ax2.set_ylim(0, 0.5)
    ax2.tick_params(axis='y', labelsize=tick_size)

    ax1.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
    ax1.set_title('相対売上と予測誤差推移 (季節性単体除外モデル)', fontsize=title_size, pad=20)

    # 凡例統合
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=legend_size, framealpha=0.8)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
