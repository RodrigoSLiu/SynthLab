def visualize_data(self, df: pd.DataFrame) -> None:
    """
    Generic visualization for synthetic datasets.

    - 1 column  -> histogram
    - 2 columns -> scatter plot
    - >2 columns -> pairwise scatter plots
    """

    if df is None or df.empty:
        raise ValueError("Empty DataFrame passed to visualization")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric columns available for visualization")

    n_cols = numeric_df.shape[1]

    # --- Single variable ---
    if n_cols == 1:
        col = numeric_df.columns[0]
        plt.figure(figsize=(6, 4))
        plt.hist(numeric_df[col], bins=50)
        plt.xlabel(col)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
        return

    # --- Two variables ---
    if n_cols == 2:
        col1, col2 = numeric_df.columns
        plt.figure(figsize=(6, 4))
        plt.scatter(numeric_df[col1], numeric_df[col2], s=10, alpha=0.5)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f"{col2} vs {col1}")
        plt.tight_layout()
        plt.show()
        return

    # --- More than two variables ---
    pairs = list(combinations(numeric_df.columns, 2))

    for col1, col2 in pairs:
        plt.figure(figsize=(6, 4))
        plt.scatter(numeric_df[col1], numeric_df[col2], s=10, alpha=0.5)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f"{col2} vs {col1}")
        plt.tight_layout()
        plt.show()