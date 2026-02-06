X = df_dataset[NODE_FEATURES]

mask = X.notna().all(axis=1)

X_clean = X.loc[mask]
# remove colunas com qualquer NaN
X_clean = X_clean.reset_index(drop=True)
meta_clean = meta_clean.reset_index(drop=True)

X_clean.shape

