import pandas as pd

df = pd.read_parquet('output/text_units.parquet')
print(f"Columns: {df.columns.tolist()}")
print(f"Has embedding column: {'embedding' in df.columns}")
if 'embedding' in df.columns:
    print(f"Embedding dimension: {len(df['embedding'].iloc[0])}")
    print(f"First few values of first embedding: {df['embedding'].iloc[0][:5]}")
else:
    print("No embedding column found.") 