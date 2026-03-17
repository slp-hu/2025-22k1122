import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# ==========================================
# 0. 表示・フォントの設定
# ==========================================
plt.rcParams["font.family"] = "MS Gothic"
font_size_global = 20
title_size = font_size_global - 5
label_size = font_size_global + 15
tick_size = font_size_global - 8
legend_size = font_size_global + 2

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\washing'
spec_file_name = 'ASIN+spec_wa.xlsx'
wa_daily_path = os.path.join(base_dir, 'wa_daily.csv')
wa_weekly_path = os.path.join(base_dir, 'wa_weekly.csv')
trends_2012_file = 'trends_wa_2012.csv' 

# ★ここを変更すれば、すべての集計・表示に反映されます
target_lags = list(range(7, 31))
lag_vars = [f'ln_y_lag{i}' for i in target_lags]

asins = list(dict.fromkeys(["B0BP6ZCQC5", "B09TSBGDN4", "B08Q7Q29CQ", "B08B1GLD8L", "B0B313BKHK",
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
           "B09DKGXYCV", "B01HUJES06", "B0162EV9QQ", "B07ZPSWGYN", "B07CDRZYM5"]))

market_main_vars = ['ln_f_sum'] 
inter_vars_m1 = ['inter_cap', 'inter_aut', 'inter_pri', 'inter_siz']
inter_vars_m2 = ['search_2012_impact']

# ==========================================
# 2. データの読み込みと前処理
# ==========================================
wa_weekly = pd.read_csv(wa_weekly_path, encoding='cp932').sort_values('date')
wa_weekly['date'] = pd.to_datetime(wa_weekly['date'])
wa_weekly = wa_weekly.set_index('date').reindex(pd.date_range('2023-01-01', '2025-12-31', freq='D')).interpolate(method='linear').reset_index().rename(columns={'index': 'date'})

wa_daily = pd.read_csv(wa_daily_path, encoding='cp932')
wa_daily['date'] = pd.to_datetime(wa_daily['date'])

t12_raw = pd.read_csv(os.path.join(base_dir, trends_2012_file), encoding='cp932')
trends_12_df = pd.DataFrame({'date': pd.to_datetime(t12_raw.iloc[:, 0]), 'search_b': t12_raw.iloc[:, 1], 'seasonal_d': t12_raw.iloc[:, 3]})

search_adj = wa_daily.merge(wa_weekly[['date', 'f_sum']], on='date', how='left').merge(trends_12_df, on='date', how='left')
num_cols_adj = search_adj.select_dtypes(include=[np.number]).columns
search_adj[num_cols_adj] = search_adj[num_cols_adj].interpolate(method='linear', limit_direction='both')

for fk in ['f_cap', 'f_aut', 'f_pri', 'f_siz']:
    search_adj[f'adj_{fk}'] = search_adj[fk] * search_adj['f_sum']

specs_master = pd.read_excel(os.path.join(base_dir, spec_file_name), usecols=[0, 1, 2, 3, 4])
specs_master.columns = ['asin', 'cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']

def get_ln(val):
    num = pd.to_numeric(val, errors='coerce')
    return np.log(float(num) + 1e-6) if (pd.notna(num) and num > 0) else 0.0

all_data = []
eps = 1e-6
for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_{asin}.csv')
    if not os.path.exists(file_path): continue
    df = pd.read_csv(file_path).assign(date=lambda x: pd.to_datetime(x['date']))
    df = df.merge(search_adj, on='date', how='left').sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    df['q_it'] = 11.74 * (df['daily_salesrank'].clip(lower=1) ** (-0.684))
    df['ln_y'] = np.log(df['q_it'] + eps)
    
    new_cols = {f'ln_y_lag{i}': df['ln_y'].shift(i) for i in target_lags}
    new_cols['ln_f_sum'] = np.log(df['f_sum'] + eps)
    
    spec = specs_master[specs_master['asin'] == asin].iloc[0] if asin in specs_master['asin'].values else None
    if spec is not None:
        new_cols['inter_cap'] = np.log(df['adj_f_cap'] + eps) * get_ln(spec['cap_spec'])
        new_cols['inter_aut'] = np.log(df['adj_f_aut'] + eps) * (pd.to_numeric(spec['aut_spec'], errors='coerce') or 0.0)
        new_cols['inter_pri'] = np.log(df['adj_f_pri'] + eps) * get_ln(spec['pri_spec'])
        new_cols['inter_siz'] = np.log(df['adj_f_siz'] + eps) * get_ln(spec['siz_spec'])
    new_cols['search_2012_impact'] = np.log(df['search_b'] + eps) * np.log(df['seasonal_d'] + eps)
    all_data.append(pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1))

full_df = pd.concat(all_data).dropna().sort_values('date')

# （中略：前処理部分までは変更なし）

# ==========================================
# 3. 逐次学習・予測ロジック
# ==========================================
def get_top_10(df, year):
    return df[df['date'].dt.year == year].groupby('asin')['q_it'].mean().nlargest(10).index.tolist()

top23, top24, top25 = get_top_10(full_df, 2023), get_top_10(full_df, 2024), get_top_10(full_df, 2025)

def run_session(train_years, test_year, train_asins, test_asins):
    # 追加：使用ASINの表示
    print(f"\n===== Session: Test Year {test_year} =====")
    print(f"学習用ASIN ({train_years}年 Top10ベース): {train_asins}")
    print(f"予測対象ASIN ({test_year}年 Top10ベース): {test_asins}")

    target_vars = lag_vars + market_main_vars + inter_vars_m1 + inter_vars_m2
    train_df = full_df[full_df['date'].dt.year.isin(train_years) & full_df['asin'].isin(train_asins)].copy()
    test_df = full_df[(full_df['date'].dt.year == test_year) & full_df['asin'].isin(test_asins)].copy()
    
    sc = StandardScaler()
    train_df[target_vars] = sc.fit_transform(train_df[target_vars])
    test_df[target_vars] = sc.transform(test_df[target_vars])
    
    dummies = pd.get_dummies(train_df['asin'], prefix='d', drop_first=True).astype(float)
    train_df = pd.concat([train_df, dummies], axis=1)
    d_cols = dummies.columns.tolist()
    
    models = {
        'M0:ベースラインモデル': lag_vars + market_main_vars + d_cols,
        'M1:製品特徴モデル': lag_vars + market_main_vars + inter_vars_m1 + d_cols,
        'M2:カテゴリ検索モデル': lag_vars + market_main_vars + inter_vars_m2 + d_cols
    }
    
    results = {}
    for name, cols in models.items():
        fit = sm.OLS(train_df['ln_y'], sm.add_constant(train_df[cols])).fit()
        ex_test = sm.add_constant(test_df, has_constant='add')
        for c in fit.params.index:
            if c not in ex_test.columns: ex_test[c] = 0
        preds = fit.predict(ex_test[fit.params.index])
        
        mae_series = np.abs(test_df['ln_y'].values - preds.values)
        asin_mae = test_df.assign(abs_err=mae_series).groupby('asin')['abs_err'].mean().to_dict()
        
        results[name] = {'mae': mean_absolute_error(test_df['ln_y'], preds), 
                         'asin_mae': asin_mae,
                         'pred': preds, 
                         'mae_series': mae_series, 
                         'params': fit.params}
    return results, test_df['date'], test_df['ln_y']

res24, d24, y24 = run_session([2023], 2024, top23, top24)
res25, d25, y25 = run_session([2023, 2024], 2025, list(set(top23+top24)), top25)

# （以下、統計表示・グラフ描画部分は変更なし）
# ==========================================
# 4. 統計表示
# ==========================================
def print_summary(year, res):
    print(f"\n--- {year}年 評価結果 ---")
    m0_k, m1_k, m2_k = 'M0:ベースラインモデル', 'M1:製品特徴モデル', 'M2:カテゴリ検索モデル'
    m0_mae, m1_mae, m2_mae = res[m0_k]['mae'], res[m1_k]['mae'], res[m2_k]['mae']
    print(f"MAE | M0: {m0_mae:.4f}, M1: {m1_mae:.4f}, M2: {m2_mae:.4f}")
    print(f"改善率 vs M0 | M1: {(1-m1_mae/m0_mae)*100:+.2f}%, M2: {(1-m2_mae/m0_mae)*100:+.2f}%")
    
    print(f"--- 個別ASIN別 MAE ---")
    test_asins = sorted(res[m0_k]['asin_mae'].keys())
    for asin in test_asins:
        a0, a1, a2 = res[m0_k]['asin_mae'][asin], res[m1_k]['asin_mae'][asin], res[m2_k]['asin_mae'][asin]
        print(f"ASIN: {asin:10s} | M0: {a0:.4f}, M1: {a1:.4f}, M2: {a2:.4f}")

    p0, p1, p2 = res[m0_k]['params'], res[m1_k]['params'], res[m2_k]['params']
    
    # 代表的な集計表示
    first_lag = lag_vars[0]
    rest_lags = lag_vars[1:]
    l_first_m0, l_first_m1, l_first_m2 = p0[first_lag], p1[first_lag], p2[first_lag]
    l_rest_m0, l_rest_m1, l_rest_m2 = p0[rest_lags].mean(), p1[rest_lags].mean(), p2[rest_lags].mean()
    min_l = target_lags[0]
    rest_label = f"Lag{target_lags[1]}-{target_lags[-1]}" if len(target_lags) > 1 else "N/A"

    print(f"[代表重み] Lag{min_l: <12} | M0: {l_first_m0:8.4f}, M1: {l_first_m1:8.4f}, M2: {l_first_m2:8.4f}")
    print(f"[代表重み] {rest_label}(Avg) | M0: {l_rest_m0:8.4f}, M1: {l_rest_m1:8.4f}, M2: {l_rest_m2:8.4f}")
    
    if 'ln_f_sum' in p0:
        print(f"[代表重み] ln_f_sum      | M0: {p0['ln_f_sum']:8.4f}, M1: {p1['ln_f_sum']:8.4f}, M2: {p2['ln_f_sum']:8.4f}")

    # --- 追加：全変数の重み（係数）一覧表示 ---
    print(f"\n<<< {year}年 全変数重み一覧 >>>")
    all_vars = p1.index.union(p2.index).union(p0.index).tolist()
    print(f"{'Variable':<25} | {'M0':>10} | {'M1':>10} | {'M2':>10}")
    print("-" * 65)
    for v in all_vars:
        v0 = f"{p0[v]:10.4f}" if v in p0 else f"{'-':>10}"
        v1 = f"{p1[v]:10.4f}" if v in p1 else f"{'-':>10}"
        v2 = f"{p2[v]:10.4f}" if v in p2 else f"{'-':>10}"
        print(f"{v:<25} | {v0} | {v1} | {v2}")
    print("-" * 65)

print_summary("2024", res24)
print_summary("2025", res25)

# ==========================================
# 5. 二軸グラフ描画
# ==========================================
y23_avg = full_df[full_df['date'].dt.year == 2023].groupby('date')['ln_y'].mean()
y24_avg = pd.DataFrame({'Actual': y24, 'date': d24}).groupby('date').mean()
y25_avg = pd.DataFrame({'Actual': y25, 'date': d25}).groupby('date').mean()
all_actual = pd.concat([y23_avg, y24_avg['Actual'], y25_avg['Actual']])

def get_mae_30d(res, dates):
    df = pd.DataFrame({
        'date': dates,
        'M0': res['M0:ベースラインモデル']['mae_series'],
        'M1': res['M1:製品特徴モデル']['mae_series'],
        'M2': res['M2:カテゴリ検索モデル']['mae_series']
    }).groupby('date').mean()
    return df.resample('30D').mean()

mae_plot = pd.concat([get_mae_30d(res24, d24), get_mae_30d(res25, d25)])

fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.plot(all_actual.index, all_actual.values, color='green', label='相対売上(平均)', linewidth=1.2, alpha=0.7)
ax1.set_ylabel('相対売上 (ln_y)', fontsize=label_size)
ax1.tick_params(axis='y', labelsize=tick_size)
ax1.tick_params(axis='x', labelsize=tick_size+14)

ax2 = ax1.twinx()
ax2.plot(mae_plot.index, mae_plot['M0'], color='gray', marker='s', markersize=4, linestyle=':', label='MAE: M0(ベースラインモデル)', alpha=0.6)
ax2.plot(mae_plot.index, mae_plot['M1'], color='blue', marker='o', markersize=5, label='MAE: M1(製品特徴モデル)', linewidth=1.5)
ax2.plot(mae_plot.index, mae_plot['M2'], color='red', marker='^', markersize=5, label='MAE: M2(カテゴリ検索モデル)', linewidth=1.2, alpha=0.8)
ax2.set_ylabel('予測誤差 (30日平均MAE)', fontsize=label_size)
ax2.tick_params(axis='y', labelsize=tick_size)

plt.axvline(pd.to_datetime('2024-01-01'), color='black', linestyle='--', alpha=0.3)
plt.axvline(pd.to_datetime('2025-01-01'), color='black', linestyle='--', alpha=0.3)
plt.title('洗濯機：相対売上とモデル別MAEの推移', fontsize=title_size)
ax1.grid(axis='y', alpha=0.2)

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1 + handler2, label1 + label2, loc='upper left', fontsize=legend_size)

plt.tight_layout()
plt.show()
