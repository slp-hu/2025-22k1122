import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import matplotlib

# ==========================================
# 0. 表示・日本語フォント設定
# ==========================================
plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "Meiryo", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False 

FONT_SIZE_BASE = 15
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': FONT_SIZE_BASE + 3,
    'axes.titlesize': FONT_SIZE_BASE + 5,
    'xtick.labelsize': FONT_SIZE_BASE - 2,
    'ytick.labelsize': FONT_SIZE_BASE - 2,
    'legend.fontsize': FONT_SIZE_BASE - 1
})

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\washing'
spec_file_name = 'ASIN+spec_wa.xlsx'
wa_daily_path = os.path.join(base_dir, 'wa_daily.csv')
wa_weekly_path = os.path.join(base_dir, 'wa_weekly.csv')
trends_2012_file = 'trends_wa_2012.csv' 

target_lags = [1]
lag_vars = ['ln_y_lag1']

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
eps = 1e-6

# ==========================================
# 2. データの準備
# ==========================================
wa_weekly = pd.read_csv(wa_weekly_path, encoding='cp932')
wa_weekly['date'] = pd.to_datetime(wa_weekly['date'])

wa_daily = pd.read_csv(wa_daily_path, encoding='cp932')
wa_daily['date'] = pd.to_datetime(wa_daily['date'])

# トレンドデータの利用可能カラムを動的に取得 (KeyError対策)
trend_cols = [c for c in ['f_sum', 'f_cap', 'f_aut', 'f_pri', 'f_siz'] if c in wa_weekly.columns]

t12_raw = pd.read_csv(os.path.join(base_dir, trends_2012_file), encoding='cp932')
trends_12_df = pd.DataFrame({'date': pd.to_datetime(t12_raw.iloc[:, 0]), 'search_b': t12_raw.iloc[:, 1], 'seasonal_d': t12_raw.iloc[:, 3]})

specs_master = pd.read_excel(os.path.join(base_dir, spec_file_name), usecols=[0, 1, 2, 3, 4])
specs_master.columns = ['asin', 'cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']

def get_ln(val):
    num = pd.to_numeric(val, errors='coerce')
    return np.log(float(num) + eps) if (pd.notna(num) and num > 0) else 0.0

# ==========================================
# 3. 解析用関数
# ==========================================
def get_top_10(df, year):
    return df[df['date'].dt.year == year].groupby('asin')['q_it'].mean().nlargest(10).index.tolist()

def run_experiment(full_df):
    top23 = get_top_10(full_df, 2023)
    top24 = get_top_10(full_df, 2024)
    top25 = get_top_10(full_df, 2025)
    
    def run_session(train_years, test_year, train_asins, test_asins):
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
            'M0': lag_vars + market_main_vars + d_cols,
            'M1': lag_vars + market_main_vars + inter_vars_m1 + d_cols,
            'M2': lag_vars + market_main_vars + inter_vars_m2 + d_cols
        }
        
        res = {}
        for name, cols in models.items():
            # 利用可能なカラムのみで学習
            available_cols = [c for c in cols if c in train_df.columns]
            fit = sm.OLS(train_df['ln_y'], sm.add_constant(train_df[available_cols])).fit()
            ex_test = sm.add_constant(test_df, has_constant='add')
            for c in fit.params.index:
                if c not in ex_test.columns: ex_test[c] = 0
            preds = fit.predict(ex_test[fit.params.index])
            res[name] = mean_absolute_error(test_df['ln_y'], preds)
        return res

    res24 = run_session([2023], 2024, top23, top24)
    res25 = run_session([2023, 2024], 2025, list(set(top23+top24)), top25)
    return res24, res25

# ==========================================
# 4. 実行
# ==========================================

final_results = []

for freq in ['W', 'MS']:
    freq_label = "週次" if freq == 'W' else "月次"
    all_data = []
    
    for asin in asins:
        file_path = os.path.join(base_dir, f'salesrank_daily_{asin}.csv')
        if not os.path.exists(file_path): continue
        
        df_d = pd.read_csv(file_path).assign(date=lambda x: pd.to_datetime(x['date']))
        
        # 外部トレンドを日次に拡張してマージ
        search_adj = wa_daily.merge(wa_weekly[['date'] + trend_cols], on='date', how='left')
        search_adj = search_adj.merge(trends_12_df, on='date', how='left')
        
        df_d = df_d.merge(search_adj, on='date', how='left').sort_values('date')
        df_d = df_d[(df_d['date'] >= '2023-01-01') & (df_d['date'] <= '2025-12-31')].reset_index(drop=True)
        
        # 数値補間
        num_cols = df_d.select_dtypes(include=[np.number]).columns
        df_d[num_cols] = df_d[num_cols].interpolate(method='linear', limit_direction='both')
        df_d['q_it'] = 11.74 * (df_d['daily_salesrank'].clip(lower=1) ** (-0.684))
        
        # リサンプリング実行
        df_d_idx = df_d.set_index('date')
        resampled = pd.DataFrame()
        resampled['q_it'] = df_d_idx['q_it'].resample(freq).sum()
        
        # 存在するカラムのみリサンプル (Error対策)
        for col in trend_cols + ['search_b', 'seasonal_d']:
            if col in df_d_idx.columns:
                resampled[col] = df_d_idx[col].resample(freq).mean()
        
        # ラグと対数変数
        resampled['ln_y'] = np.log(resampled['q_it'] + eps)
        resampled['ln_y_lag1'] = resampled['ln_y'].shift(1)
        resampled['ln_f_sum'] = np.log(resampled.get('f_sum', pd.Series(0, index=resampled.index)) + eps)
        
        # 相互作用変数 (カラムの存在をチェック)
        spec = specs_master[specs_master['asin'] == asin].iloc[0] if asin in specs_master['asin'].values else None
        if spec is not None:
            f_sum = resampled.get('f_sum', 1.0)
            resampled['inter_cap'] = np.log(resampled.get('f_cap', 0) * f_sum + eps) * get_ln(spec['cap_spec'])
            resampled['inter_aut'] = np.log(resampled.get('f_aut', 0) * f_sum + eps) * (pd.to_numeric(spec['aut_spec'], errors='coerce') or 0.0)
            resampled['inter_pri'] = np.log(resampled.get('f_pri', 0) * f_sum + eps) * get_ln(spec['pri_spec'])
            resampled['inter_siz'] = np.log(resampled.get('f_siz', 0) * f_sum + eps) * get_ln(spec['siz_spec'])
        
        resampled['search_2012_impact'] = np.log(resampled.get('search_b', 0) + eps) * np.log(resampled.get('seasonal_d', 0) + eps)
        resampled['asin'] = asin
        all_data.append(resampled.reset_index())

    if not all_data: continue
    full_df = pd.concat(all_data).dropna()
    res24, res25 = run_experiment(full_df)
    
    for year, res in [("2024", res24), ("2025", res25)]:
        final_results.append({
            '分析単位': freq_label, '対象年': year,
            'M0_MAE': res['M0'], 'M1_MAE': res['M1'], 'M2_MAE': res['M2'],
            'M1改善率(%)': (1 - res['M1']/res['M0']) * 100,
            'M2改善率(%)': (1 - res['M2']/res['M0']) * 100
        })

# ==========================================
# 5. 出力
# ==========================================
if final_results:
    summary_df = pd.DataFrame(final_results)
    print("\n===== 洗濯機：精度比較サマリー (Lag1) =====")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    plot_data = summary_df.set_index(['分析単位', '対象年'])[['M1改善率(%)', 'M2改善率(%)']]
    plot_data.plot(kind='bar', figsize=(12, 7))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('洗濯機：分析単位別改善率比較')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
