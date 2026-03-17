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
font_size_global = 12
plt.rcParams.update({'font.size': font_size_global, 'axes.titlesize': 14, 'axes.labelsize': 12})

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
market_main_vars = ['ln_f_sum']
inter_vars = ['inter_cap', 'inter_aut', 'inter_pri', 'inter_siz']
target_vars = lag_vars + market_main_vars + inter_vars

# ==========================================
# 2. データ読み込み・検索トレンド構築
# ==========================================
wa_weekly = pd.read_csv(wa_weekly_path)
wa_weekly['date'] = pd.to_datetime(wa_weekly['date'])
wa_weekly = wa_weekly.sort_values('date').set_index('date')

daily_range = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
wa_weekly_interp = wa_weekly.reindex(daily_range).interpolate(method='linear').reset_index().rename(columns={'index': 'date'})

wa_daily = pd.read_csv(wa_daily_path)
wa_daily['date'] = pd.to_datetime(wa_daily['date'])

search_adj = wa_daily.merge(wa_weekly_interp[['date', 'f_sum']], on='date', how='left')
for fk in ['f_cap', 'f_aut', 'f_pri', 'f_siz']:
    search_adj[f'adj_{fk}'] = search_adj[fk] * search_adj['f_sum']

search_final = search_adj[['date', 'adj_f_cap', 'adj_f_aut', 'adj_f_pri', 'adj_f_siz', 'f_sum']]

# ==========================================
# 3. データ統合と前処理
# ==========================================
specs_master = pd.read_excel(spec_file_path, usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']

def get_numeric_scalar(val):
    num = pd.to_numeric(val, errors='coerce')
    return float(num) if pd.notna(num) else 0.0

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
    df['daily_salesrank'] = df['daily_salesrank'].interpolate(method='linear', limit_direction='both')
    
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec_row = spec_match.iloc[0]
    
    # 相対売上の算出
    k_c, A_c = 0.713, 1.0
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it'] + eps)

    new_cols = {}
    for i in range(1, max_lag + 1):
        new_cols[f'ln_y_lag{i}'] = df['ln_y'].shift(i)
    
    new_cols['ln_f_sum'] = np.log(df['f_sum'] + eps)
    
    ln_z_cap = np.log(df['adj_f_cap'] + eps)
    ln_z_aut = np.log(df['adj_f_aut'] + eps)
    ln_z_pri = np.log(df['adj_f_pri'] + eps)
    ln_z_siz = np.log(df['adj_f_siz'] + eps)

    f_cap_s = np.log(get_numeric_scalar(spec_row['cap_spec']) + eps)
    f_pri_s = np.log(get_numeric_scalar(spec_row['pri_spec']) + eps)
    f_siz_s = np.log(get_numeric_scalar(spec_row['siz_spec']) + eps)
    d_aut_s = get_numeric_scalar(spec_row['aut_spec']) 

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

model_p = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[target_vars + dummy_cols])).fit()
model_b = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[lag_vars + dummy_cols])).fit()

# --- 係数出力 ---
print("\n【推定係数一覧】")
display_vars = lag_vars + market_main_vars + inter_vars
weights_df = pd.DataFrame(model_p.params[display_vars], columns=['Coefficient'])
pd.set_option('display.max_rows', None)
print(weights_df.to_string(formatters={'Coefficient':'{:,.4f}'.format}))
pd.reset_option('display.max_rows')

def get_aligned_exog(batch_df, model, columns):
    exog = sm.add_constant(batch_df[columns], has_constant='add')
    for c in model.params.index:
        if c not in exog.columns: exog[c] = 0.0
    return exog[model.params.index]

mae_results = []
current_date = test_overall['date'].min()
while current_date <= test_overall['date'].max():
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= current_date + pd.Timedelta(days=29))]
    if not batch.empty:
        pred_p = model_p.predict(get_aligned_exog(batch, model_p, target_vars + dummy_cols))
        pred_b = model_b.predict(get_aligned_exog(batch, model_b, lag_vars + dummy_cols))
        m_p, m_b = mean_absolute_error(batch['ln_y'], pred_p), mean_absolute_error(batch['ln_y'], pred_b)
        mae_results.append({'テスト開始': current_date, 'MAE_ベースライン': m_b, 'MAE_提案モデル': m_p})
    current_date += pd.Timedelta(days=30)

err_df = pd.DataFrame(mae_results)

# ==========================================
# 5. 可視化 (result_v3.png スタイル)
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

# 背面：日次 ln_y (全製品の平均)
daily_lny_avg = full_df.groupby('date')['ln_y'].mean().reset_index()
ax1.plot(daily_lny_avg['date'], daily_lny_avg['ln_y'], color='#FFCCCC', linewidth=0.8, label='相対売上', alpha=0.8)
ax1.set_xlabel('日付', fontsize=14)
ax1.set_ylabel('相対売上（ln_y）', fontsize=14)
ax1.set_ylim(daily_lny_avg['ln_y'].min() - 0.2, daily_lny_avg['ln_y'].max() + 0.2)

# 右軸：予測誤差 (MAE)
ax2 = ax1.twinx()
ax2.plot(err_df['テスト開始'], err_df['MAE_提案モデル'], color='#D62728', marker='o', markersize=4, label='MAE：提案モデル', linewidth=1.5)
ax2.plot(err_df['テスト開始'], err_df['MAE_ベースライン'], color='#1F77B4', marker='s', markersize=4, label='MAE：ベースライン', linewidth=1.5)
ax2.set_ylabel('予測誤差（MAE）', fontsize=14)
ax2.set_ylim(0, 0.5)  # result_v3に準拠

# グリッド・目盛設定
ax1.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
ax1.set_title('相対売上と予測誤差推移', fontsize=15, pad=15)

# 凡例統合
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.show()
