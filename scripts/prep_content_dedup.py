"""อ่าน xlsx เอาเฉพาะ content, ตัด dup, เซฟ CSV"""
import pandas as pd

df = pd.read_excel("data/alltime_25_26.xlsx", usecols=["content"])
df = df.drop_duplicates(subset="content").reset_index(drop=True)
out = "data/alltime_25_26_content_dedup.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(df):,} rows -> {out}")
