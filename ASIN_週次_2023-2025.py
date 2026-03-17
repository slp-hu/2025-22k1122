import os
import sys
import keepa
import pandas as pd
import numpy as np
import time

# ========= 設定 =========
YOUR_API_KEY = "imukja73bgdmvbbp14ecut9n6sggdtlqqpvth9vps3ksqa7eerplb1j48dt5gdl7"

# ご提示いただいた新しいASINリスト
asin_list = [
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

OUT_DIR = r"C:\Users\admin\Desktop\4年\本実験\ASIN\washing"

# 集計設定
WEEK_AGG = "min"  # その週の最高順位（数値が最小）を取得
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# ========= 実行 =========
print("--- Keepa SalesRank 週次集計 (1/1起点・欠損値は空欄) ---")

try:
    api = keepa.Keepa(YOUR_API_KEY)
except Exception as e:
    print(f"API初期化エラー: {e}")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

for i, asin in enumerate(asin_list):
    asin = str(asin).strip()
    print(f"[{i+1}/{len(asin_list)}] 取得中: {asin} ...")
    
    out_csv = os.path.join(OUT_DIR, f"salesrank_weekly_{asin}.csv")
    if os.path.exists(out_csv):
        print(f"  -> スキップ (既に存在します)")
        continue

    for attempt in range(3):
        try:
            products = api.query(asin, domain="JP", history=True, progress_bar=False)
            
            if not products or "data" not in products[0]:
                print(f"  -> データなし: {asin}")
                break
                
            p = products[0]
            sales_vals = p.get("data", {}).get("SALES")
            sales_time = p.get("data", {}).get("SALES_time")

            if sales_vals is None or len(sales_vals) == 0:
                print(f"  -> SalesRank履歴なし: {asin}")
                break

            # 時刻正規化
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
            
            # 順位が0以下のものは異常値として除外（欠損値にする）
            df.loc[df["sales_rank"] <= 0, "sales_rank"] = np.nan

            # 期間抽出
            start = pd.Timestamp(f"{START_DATE} 00:00:00", tz="Asia/Tokyo")
            end = pd.Timestamp(f"{END_DATE} 23:59:59", tz="Asia/Tokyo")
            
            # 週次の基準枠を作成（1/1, 1/8...）
            date_range = pd.date_range(start=start, end=end, freq='7D', tz="Asia/Tokyo")

            # 1/1起点で集計
            weekly_df = df.resample("7D", origin=start, label='left').agg(WEEK_AGG)
            
            # 指定期間の全週が含まれるように再配置（データがない週を空にする）
            weekly_df = weekly_df.reindex(date_range)
            
            # 整形
            out = weekly_df.reset_index().rename(columns={"index": "week_start_date", "sales_rank": "weekly_salesrank"})
            
            # Int64型を使うことで、整数を維持したまま欠損値を許容
            out["weekly_salesrank"] = out["weekly_salesrank"].astype(float).round().astype("Int64")
            
            out.insert(0, "asin", asin)
            out["week_start_date"] = out["week_start_date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d")

            # 保存
            out.to_csv(out_csv, index=False, encoding="utf-8-sig", na_rep="")
            print(f"  ✅ 週次保存完了: {len(out)}週分")
            break

        except Exception as e:
            if attempt < 2:
                print(f"  ⚠️ 接続エラー再試行中... ({attempt+1}/3)")
                time.sleep(5)
            else:
                print(f"  ❌ エラー: {e}")
                
    time.sleep(2)  # API負荷軽減のための待機

print("\n--- 全ての処理が完了しました ---")
