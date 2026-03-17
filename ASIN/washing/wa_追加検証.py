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
plt.rcParams.update({'font.size': 12})

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\washing'
spec_file_name = 'ASIN+spec_wa.xlsx'
wa_daily_path = os.path.join(base_dir, 'wa_daily.csv')
wa_weekly_path = os.path.join(base_dir, 'wa_weekly.csv')
trends_2012_file = 'trends_wa_2012.csv' 

# 実験のパラメータ
max_lag = 30
start_lag_range = list(range(2, 15)) # 2から14まで開始位置を振る

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
    
    # 1～30の全ラグを生成
    lags_df = pd.DataFrame({f'ln_y_lag{i}': df['ln_y'].shift(i) for i in range(1, max_lag + 1)})
    
    new_cols_basic = pd.DataFrame()
    new_cols_basic['ln_f_sum'] = np.log(df['f_sum'] + eps)
    new_cols_basic['search_2012_impact'] = np.log(df['search_b'] + eps) * np.log(df['seasonal_d'] + eps)
    
    spec = specs_master[specs_master['asin'] == asin].iloc[0] if asin in specs_master['asin'].values else None
    if spec is not None:
        new_cols_basic['inter_cap'] = np.log(df['adj_f_cap'] + eps) * get_ln(spec['cap_spec'])
        new_cols_basic['inter_aut'] = np.log(df['adj_f_aut'] + eps) * (pd.to_numeric(spec['aut_spec'], errors='coerce') or 0.0)
        new_cols_basic['inter_pri'] = np.log(df['adj_f_pri'] + eps) * get_ln(spec['pri_spec'])
        new_cols_basic['inter_siz'] = np.log(df['adj_f_siz'] + eps) * get_ln(spec['siz_spec'])
        
    all_data.append(pd.concat([df, lags_df, new_cols_basic], axis=1))

# 条件を揃えるため一括でdropna
full_df_master = pd.concat(all_data).dropna().sort_values('date')

# ==========================================
# 3. 実験実行用関数
# ==========================================
def run_experiment(start_lag, train_years, test_year, train_asins, test_asins):
    current_lags = [f'ln_y_lag{i}' for i in range(start_lag, max_lag + 1)]
    target_vars = current_lags + market_main_vars + inter_vars_m1 + inter_vars_m2
    
    train_df = full_df_master[full_df_master['date'].dt.year.isin(train_years) & full_df_master['asin'].isin(train_asins)].copy()
    test_df = full_df_master[(full_df_master['date'].dt.year == test_year) & full_df_master['asin'].isin(test_asins)].copy()
    
    sc = StandardScaler()
    train_df[target_vars] = sc.fit_transform(train_df[target_vars])
    test_df[target_vars] = sc.transform(test_df[target_vars])
    
    dummies = pd.get_dummies(train_df['asin'], prefix='d', drop_first=True).astype(float)
    train_df = pd.concat([train_df, dummies], axis=1)
    d_cols = dummies.columns.tolist()
    
    models = {
        'M0': current_lags + market_main_vars + d_cols,
        'M1': current_lags + market_main_vars + inter_vars_m1 + d_cols,
        'M2': current_lags + market_main_vars + inter_vars_m2 + d_cols
    }
    
    summary = {}
    for name, cols in models.items():
        fit = sm.OLS(train_df['ln_y'], sm.add_constant(train_df[cols])).fit()
        ex_test = sm.add_constant(test_df, has_constant='add')
        for c in fit.params.index:
            if c not in ex_test.columns: ex_test[c] = 0
        preds = fit.predict(ex_test[fit.params.index])
        
        summary[name] = {
            'mae': mean_absolute_error(test_df['ln_y'], preds),
            'params': fit.params
        }
    
    m0_mae = summary['M0']['mae']
    m1_mae, m2_mae = summary['M1']['mae'], summary['M2']['mae']
    f_lag = current_lags[0]
    
    # 指標の計算
    m1_imp = (1 - m1_mae/m0_mae) * 100
    m2_imp = (1 - m2_mae/m0_mae) * 100
    m1_ratio = (summary['M1']['params'][inter_vars_m1].abs().sum() / abs(summary['M1']['params'][f_lag])) * 100
    m2_ratio = (summary['M2']['params'][inter_vars_m2].abs().sum() / abs(summary['M2']['params'][f_lag])) * 100
    
    return {
        'StartLag': start_lag,
        'M1_Imp%': m1_imp,
        'M2_Imp%': m2_imp,
        'AvgProposedMAE': (m1_mae + m2_mae) / 2,
        'M1_WeightRatio%': m1_ratio,
        'M2_WeightRatio%': m2_ratio,
        'AR1_Weight': summary['M1']['params'][f_lag],
        'OtherLagsAvg': summary['M1']['params'][current_lags[1:]].mean() if len(current_lags) > 1 else 0
    }

# ==========================================
# 4. 一括ループ実行とテーブル表示
# ==========================================
def get_top_10(df, year):
    return df[df['date'].dt.year == year].groupby('asin')['q_it'].mean().nlargest(10).index.tolist()

top23, top24, top25 = get_top_10(full_df_master, 2023), get_top_10(full_df_master, 2024), get_top_10(full_df_master, 2025)

for test_year, train_yrs, t_asins, v_asins in [(2024, [2023], top23, top24), (2025, [2023, 2024], list(set(top23+top24)), top25)]:
    results = []
    print(f"\n>>> 洗濯機 {test_year}年 実験実行中...")
    for s in start_lag_range:
        results.append(run_experiment(s, train_yrs, test_year, t_asins, v_asins))
    
    res_df = pd.DataFrame(results).set_index('StartLag')
    
    print(f"\n--- {test_year}年 洗濯機ラグ別サマリー ---")
    display_cols = ['M1_Imp%', 'M2_Imp%', 'AvgProposedMAE', 'M1_WeightRatio%', 'M2_WeightRatio%']
    print(res_df[display_cols].round(4))

    # --- 修正後のグラフ描画部分 ---
    plt.figure(figsize=(10, 5))
    # 'M1_Imp%' と 'M2_Imp%' を正しく指定
    plt.plot(res_df.index, res_df['M1_Imp%'], marker='o', label='M1(製品特徴) 改善率')
    plt.plot(res_df.index, res_df['M2_Imp%'], marker='s', label='M2(カテゴリ) 改善率') 
    
    plt.title(f'洗濯機 {test_year}: ラグ開始位置と予測精度の関係')
    plt.xlabel('ラグ開始位置 (上限30固定)')
    plt.ylabel('MAE改善率 (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    #plt.show()
