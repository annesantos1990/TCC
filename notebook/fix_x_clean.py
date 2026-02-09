
# Fix for ValueError: Found array with 0 sample(s)
# The error occurs because the filtering condition removes all rows.
# mask = X.notna().all(axis=1) checks if a row has NO NaNs across ALL 366 columns.
# If every subject has at least one missing local metric, mask is all False.

# Solution: Fill NaNs with 0 (as done in the earlier analysis) instead of dropping rows.

# Assuming df_dataset is your main DataFrame
X = df_dataset[NODE_FEATURES]

# Check how many NaNs we have (for information)
print(f"Total NaNs: {X.isna().sum().sum()}")
print(f"Rows with at least one NaN: {(X.isna().any(axis=1)).sum()} out of {len(X)}")

# Fill NaNs with 0
X_clean = X.fillna(0)

# If you have a metadata dataframe (meta_clean), ensure it is aligned.
# Since we didn't drop any rows from X, we don't need to drop rows from meta_clean
if 'meta_clean' in locals():
    # Just ensure it matches X index-wise if needed, but if it came from df_dataset it should be fine.
    pass
else:
    # If meta_clean wasn't defined yet, define it likely from df_dataset
    # Adjust the column selection as needed based on your notebook
    meta_cols = ["subject_id", "condition", "age", "gender"] # Example columns
    available_meta = [c for c in meta_cols if c in df_dataset.columns]
    meta_clean = df_dataset[available_meta]

print(f"X_clean shape: {X_clean.shape}")

# Now you can scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

print("Scaling successful.")
