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

# ファイルパスの定義
wa_weekly_path = os.path.join(base_dir, 'wa_weekly.csv')
spec_file_path = os.path.join(base_dir, spec_file_name)

# 週次分析：ラグ数4（1ヶ月分）
max_lag = 4

asins = [
    "B0BP6ZCQC5", "B09TSBGDN4", "B08Q7Q29CQ", "B08B1GLD8L", "B0B313BKHK",
    "B0B31621RB", "B084K7PM6J", "B07JR174ZT", "B09J8MF636", "B0BQQKTMVD",
    "B07MZLN15F", "B07RWR7BX6", "B08T1DM8KQ", "B0815PXPKZ", "B08KH6QFYZ",
    "B01C10VSM0", "B08B13BXQL", "B094VFZXMG", "B01G58K3KM", "B0BP6ZS2TX",
    "B0BP6VPD4K", "B09N71PGTB", "B09N73WQ2V", "B0BD3GJGLF", "B0BD3FYJ3S",
    "B0BH8JRJCQ", "B075B5WR85", "B083N9LZ6Y", "B0BP6J3SQX", "B07LBWQ3BF",
    "B01C10VTE2", "B07YLRMKRG", "B07JQX29XR", "B0BLMJJ9TV", "B09DKK6VCL",
    "B09P31T6GF", "B083PR3Y2K", "B07KCNJY79", "B09DSQQGT8", "B07SQCZRCG",
    "B09DKKZHFC", "B07YC9Z3N1", "B07YCBRJFQ", "B07YC9Z3MV", "B09NNH112Q",
    "B0956GJ3CL", "B0BPM6K84W", "B0BPM8TQLZ", "B09571SNLK", "B0956PSK2H",
    "B0956MW64T", "B09G2L68H2", "B09G2KS69R", "B09G2LJQ98", "B09G2LNCXG",
    "B09DKGXYCV", "B01HUJES06", "B0162EV9QQ", "B07ZPSWGYN", "B07CDRZYM5"
]

# 変数リスト
lag_vars = [f'ln_y_lag{i}' for i in range(1, max_lag + 1)]
spec_vars = ['ln_f_cap', 'f_aut_raw', 'ln_f_pri', 'ln_f_siz']
inter_vars = ['inter_cap', 'inter_aut', 'inter_pri', 'inter_siz']
target_vars = lag_vars + spec_vars + inter_vars

# ==========================================
# 2. 週次検索データの読み込み
# ==========================================
if not os.path.exists(wa_weekly_path):
    raise FileNotFoundError(f"wa_weekly.csv が見つかりません: {wa_weekly_path}")

wa_weekly = pd.read_csv(wa_weekly_path)
wa_weekly['date'] = pd.to_datetime(wa_weekly['date'])
wa_weekly = wa_weekly.sort_values('date')

# ==========================================
# 3. スペックマスタと週次SRデータの統合
# ==========================================
if not os.path.exists(spec_file_path):
    raise FileNotFoundError(f"スペックファイルが見つかりません: {spec_file_path}")

specs_master = pd.read_excel(spec_file_path, usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']

all_data = []
for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_weekly_{asin}.csv')
    if not os.path.exists(file_path): 
        continue
    
    df_weekly = pd.read_csv(file_path)
    # カラム名の正規化
    if 'week_start_date' in df_weekly.columns:
        df_weekly = df_weekly.rename(columns={'week_start_date': 'date'})
    df_weekly['date'] = pd.to_datetime(df_weekly['date'])
    
    # 検索データ(wa_weekly)をマージ
    df_weekly = df_weekly.merge(wa_weekly, on='date', how='inner')
    df_weekly = df_weekly.sort_values('date')
    df_weekly = df_weekly[(df_weekly['date'] >= '2023-01-01') & (df_weekly['date'] <= '2025-12-31')]
    
    if len(df_weekly) < max_lag + 1: continue

    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec_row = spec_match.iloc[0]
    for col in ['cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']: 
        df_weekly[col] = spec_row[col]
    
    # 週次売上の算出
    k_c, A_c = 0.713, 1.0
    df_weekly['q_it'] = A_c * (df_weekly['weekly_salesrank'].clip(lower=1) ** (-k_c))
    df_weekly['ln_y'] = np.log(df_weekly['q_it'] * df_weekly['f_sum'] + 1e-6)

    new_cols = {}
    for i in range(1, max_lag + 1):
        new_cols[f'ln_y_lag{i}'] = df_weekly['ln_y'].shift(i)
    
    eps = 1e-6
    # スペック・検索ボリュームの処理
    new_cols['ln_f_cap'] = np.log(pd.to_numeric(df_weekly['cap_spec'], errors='coerce').fillna(0) + eps)
    new_cols['ln_z_cap'] = np.log(pd.to_numeric(df_weekly['f_cap'], errors='coerce').fillna(0) + eps)
    
    new_cols['f_aut_raw'] = pd.to_numeric(df_weekly['aut_spec'], errors='coerce').fillna(0)
    new_cols['ln_z_aut'] = np.log(pd.to_numeric(df_weekly['f_aut'], errors='coerce').fillna(0) + eps)
    
    new_cols['ln_f_pri'] = np.log(pd.to_numeric(df_weekly['pri_spec'], errors='coerce').fillna(0) + eps)
    new_cols['ln_z_pri'] = np.log(pd.to_numeric(df_weekly['f_pri'], errors='coerce').fillna(0) + eps)
    
    new_cols['ln_f_siz'] = np.log(pd.to_numeric(df_weekly['siz_spec'], errors='coerce').fillna(0) + eps)
    new_cols['ln_z_siz'] = np.log(pd.to_numeric(df_weekly['f_siz'], errors='coerce').fillna(0) + eps)

    df_weekly = pd.concat([df_weekly, pd.DataFrame(new_cols, index=df_weekly.index)], axis=1)
    
    # 交差項の作成
    df_weekly['inter_cap'] = df_weekly['ln_z_cap'] * df_weekly['ln_f_cap']
    df_weekly['inter_aut'] = df_weekly['ln_z_aut'] * df_weekly['f_aut_raw']
    df_weekly['inter_pri'] = df_weekly['ln_z_pri'] * df_weekly['ln_f_pri']
    df_weekly['inter_siz'] = df_weekly['ln_z_siz'] * df_weekly['ln_f_siz']
        
    all_data.append(df_weekly)

full_df = pd.concat(all_data).dropna().sort_values('date')
asin_dummies = pd.get_dummies(full_df['asin'], drop_first=True).astype(float)
full_df = pd.concat([full_df, asin_dummies], axis=1)

# ==========================================
# 4. 学習と評価
# ==========================================
train_data = full_df[full_df['date'].dt.year == 2023].copy()
test_overall = full_df[full_df['date'].dt.year >= 2024].copy()

dummy_cols = asin_dummies.columns.tolist()
X_proposed_cols = target_vars + dummy_cols
X_baseline_cols = lag_vars + dummy_cols

# 回帰実行
model_p = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_proposed_cols])).fit()
model_b = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_baseline_cols])).fit()

def get_aligned_exog(batch_df, model):
    exog = sm.add_constant(batch_df, has_constant='add')
    model_features = model.params.index.tolist()
    for col in model_features:
        if col not in exog.columns: exog[col] = 0.0
    return exog[model_features]

# 4週間ごと評価
mae_results = []
current_date = test_overall['date'].min()
end_date = test_overall['date'].max()

while current_date <= end_date:
    window_end = current_date + pd.Timedelta(days=27) 
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= window_end)]
    if not batch.empty:
        exog_p = get_aligned_exog(batch[X_proposed_cols], model_p)
        exog_b = get_aligned_exog(batch[X_baseline_cols], model_b)
        pred_p, pred_b = model_p.predict(exog_p), model_b.predict(exog_b)
        m_p, m_b = mean_absolute_error(batch['ln_y'], pred_p), mean_absolute_error(batch['ln_y'], pred_b)
        mae_results.append({'テスト開始': current_date.strftime('%Y-%m-%d'), 'MAE_ベースライン': m_b, 'MAE_提案モデル': m_p, '改善率(%)': (m_b - m_p) / m_b * 100 if m_b != 0 else 0})
    current_date += pd.Timedelta(days=28)

# ==========================================
# 5. 結果表示
# ==========================================
err_df = pd.DataFrame(mae_results)
print(err_df.to_string(index=True))

print("\n========================================")
print("提案モデルの主要な変数における重み（回帰係数）")
print("========================================")
# ★修正：KeyErrorを防ぐため、存在するラベルのみを安全に抽出
drop_labels = ['const'] + dummy_cols
main_params = model_p.params.drop(labels=[l for l in drop_labels if l in model_p.params.index])
print(main_params)

print(f"\n週次分析 全期間の平均改善率: {err_df['改善率(%)'].mean():.2f}%")

plt.figure(figsize=(16, 9))
plt.plot(pd.to_datetime(err_df['テスト開始']), err_df['MAE_提案モデル'], label='MAE: 提案モデル', color='tab:red', marker='o')
plt.plot(pd.to_datetime(err_df['テスト開始']), err_df['MAE_ベースライン'], label='MAE: ベースライン', color='tab:blue', marker='s')
plt.title('洗濯機：週次予測精度推移 (エラー修正済み)', fontsize=title_size)
plt.xlabel('テスト開始週', fontsize=label_size)
plt.ylabel('MAE', fontsize=label_size)
plt.legend(fontsize=legend_size)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
