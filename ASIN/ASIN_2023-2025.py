import os
import sys
import keepa
import pandas as pd
import numpy as np
import time

# ========= 設定 =========
# APIキー
YOUR_API_KEY = "imukja73bgdmvbbp14ecut9n6sggdtlqqpvth9vps3ksqa7eerplb1j48dt5gdl7"

# 直接指定するASINリスト
asin_list = [
    "B083SP2KX8"
]

# ファイルの出力先ディレクトリ
OUT_DIR = r"C:\Users\admin\Desktop\4年\本実験\ASIN"

# 日次集計方法 ("min"=最良順位を採用)
DAY_AGG = "min"

# 取得期間
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# ========= 実行 =========
print("--- Keepa SalesRank 複数ASIN一括取得 (指定リスト) ---")

# APIクライアントを初期化
try:
    api = keepa.Keepa(YOUR_API_KEY)
except Exception as e:
    print(f"API初期化エラー: {e}")
    sys.exit(1)

# 出力ディレクトリの作成
os.makedirs(OUT_DIR, exist_ok=True)
print(f"対象ASIN数: {len(asin_list)}件")

# 各ASINに対してループ処理
for i, asin in enumerate(asin_list):
    asin = str(asin).strip()
    print(f"[{i+1}/{len(asin_list)}] 取得中: {asin} ...")
    
    out_csv = os.path.join(OUT_DIR, f"salesrank_daily_2023_2025_{asin}.csv")
    
    if os.path.exists(out_csv):
        print(f"  -> スキップ (既に存在します)")
        continue

    try:
        # Keepa API 問い合わせ
        products = api.query(asin, domain="JP", history=True, progress_bar=False)
        
        if not products or "data" not in products[0]:
            print(f"  -> データなし: {asin}")
            continue
            
        p = products[0]
        sales_vals = p.get("data", {}).get("SALES")
        sales_time = p.get("data", {}).get("SALES_time")

        if sales_vals is None or len(sales_vals) == 0:
            print(f"  -> SalesRank履歴なし: {asin}")
            continue

        # 時刻正規化のロジック
        first_time = sales_time[0]
        if isinstance(first_time, (int, np.integer, float)):
            arr = np.array(sales_time)
            base = pd.Timestamp("2011-01-01", tz="UTC")
            ts_utc = base + pd.to_timedelta(arr.astype("int64"), unit="m")
        else:
            ts_utc = pd.to_datetime(sales_time, utc=True)

        # DataFrame作成
        ts_jst = ts_utc.tz_convert("Asia/Tokyo")
        df = pd.DataFrame({"sales_rank": pd.to_numeric(sales_vals, errors="coerce")}, index=ts_jst)

        # クレンジング
        df.loc[df["sales_rank"] <= 0, "sales_rank"] = pd.NA
        df = df.dropna()

        # 期間抽出
        start = pd.Timestamp(f"{START_DATE} 00:00:00", tz="Asia/Tokyo")
        end = pd.Timestamp(f"{END_DATE} 23:59:59", tz="Asia/Tokyo")
        df_target = df.loc[(df.index >= start) & (df.index <= end)]

        if df_target.empty:
            print(f"  -> 指定期間内にデータなし")
            continue

        # 日次サンプリング
        date_range = pd.date_range(start=start.normalize(), end=end.normalize(), freq='D', tz="Asia/Tokyo")
        daily_indexed = df_target.resample("D").agg(DAY_AGG).reindex(date_range)
        
        # 整形
        out = daily_indexed.reset_index().rename(columns={"index": "date", "sales_rank": "daily_salesrank"})
        out["daily_salesrank"] = out["daily_salesrank"].round().astype("Int64")
        out.insert(0, "asin", asin)
        out["date"] = out["date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d")

        # 保存
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  ✅ 保存完了: {len(out)}日分")

        # API負荷軽減のための待機
        time.sleep(1)

    except Exception as e:
        print(f"  ❌ エラー発生 ({asin}): {e}")
        continue

print("\n--- 全ての処理が完了しました ---")
