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
# 2. データ読み込み・前処理 (変更なし)
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
    df = df.merge(search_final, on='date', how='left').sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    df['daily_salesrank'] = df['daily_salesrank'].interpolate(method='linear', limit_direction='both')
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec_row = spec_match.iloc[0]
    k_c, A_c = 0.713, 1.0
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it'] + eps)
    new_cols = {f'ln_y_lag{i}': df['ln_y'].shift(i) for i in range(1, max_lag + 1)}
    new_cols['ln_f_sum'] = np.log(df['f_sum'] + eps)
    for k, s_name in zip(['cap', 'aut', 'pri', 'siz'], ['cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']):
        ln_z = np.log(df[f'adj_f_{k}'] + eps)
        s_val = get_numeric_scalar(spec_row[s_name])
        s_val = np.log(s_val + eps) if k != 'aut' else s_val
        new_cols[f'inter_{k}'] = ln_z * s_val
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df['asin'] = asin
    all_data.append(df)
full_df = pd.concat(all_data).dropna().sort_values('date')

# ==========================================
# 4. ローリングウィンドウ方式（重み保存付き）
# ==========================================
mae_results = []
weights_history = []  # 係数の履歴を保存
window_size_days = 365
forecast_size_days = 30
current_forecast_start = pd.to_datetime('2024-01-01')
end_date_limit = full_df['date'].max()

while current_forecast_start + pd.Timedelta(days=forecast_size_days) <= end_date_limit:
    train_end = current_forecast_start - pd.Timedelta(days=1)
    train_start = current_forecast_start - pd.Timedelta(days=window_size_days)
    
    train_sub = full_df[(full_df['date'] >= train_start) & (full_df['date'] <= train_end)].copy()
    test_sub = full_df[(full_df['date'] >= current_forecast_start) & (full_df['date'] <= current_forecast_start + pd.Timedelta(days=forecast_size_days-1))].copy()
    
    if train_sub.empty or test_sub.empty:
        current_forecast_start += pd.Timedelta(days=forecast_size_days)
        continue

    # ダミー変数処理
    train_dummies = pd.get_dummies(train_sub['asin'], drop_first=True).astype(float)
    X_train_p = sm.add_constant(pd.concat([train_sub[target_vars], train_dummies], axis=1))
    
    # モデル推定
    model_p = sm.OLS(train_sub['ln_y'], X_train_p).fit()
    
    # 重みの保存（定数項、ラグ変数、主要変数のみ）
    current_weights = model_p.params[target_vars].to_dict()
    current_weights['date'] = current_forecast_start
    weights_history.append(current_weights)

    # 予測
    test_dummies = pd.get_dummies(test_sub['asin'], drop_first=True).astype(float)
    for col in train_dummies.columns:
        if col not in test_dummies.columns: test_dummies[col] = 0.0
    X_test_p = sm.add_constant(pd.concat([test_sub[target_vars], test_dummies[train_dummies.columns]], axis=1), has_constant='add')
    
    pred_p = model_p.predict(X_test_p)
    mae_results.append({'date': current_forecast_start, 'MAE': mean_absolute_error(test_sub['ln_y'], pred_p)})
    current_forecast_start += pd.Timedelta(days=forecast_size_days)

# ==========================================
# 5. 推定結果の出力
# ==========================================
weights_df = pd.DataFrame(weights_history).set_index('date')

print("\n【推定係数（全期間平均）】")
# 全ローリング期間の平均係数を算出
mean_weights = weights_df.mean().to_frame(name='Average Coefficient')
print(mean_weights.to_string(formatters={'Average Coefficient':'{:,.4f}'.format}))

print("\n【予測精度(MAE)平均】")
print(f"提案モデル MAE: {pd.DataFrame(mae_results)['MAE'].mean():.4f}")

# ==========================================
# 6. 重みの推移の可視化
# ==========================================
plt.figure(figsize=(12, 6))
# 主要な変数（検索トレンドと交差項）のみプロット
focus_vars = market_main_vars + inter_vars
for var in focus_vars:
    plt.plot(weights_df.index, weights_df[var], marker='o', label=var)

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('主要説明変数の推定係数（重み）の推移', fontsize=15)
plt.xlabel('テスト開始時期', fontsize=12)
plt.ylabel('係数の値', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
