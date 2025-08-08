def pretty_info(df, name):
    print(f"{name}:")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rows: {df.shape[0]}")
    print("-" * 50)
