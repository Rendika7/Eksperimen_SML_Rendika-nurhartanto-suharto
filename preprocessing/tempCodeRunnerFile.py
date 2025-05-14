    # Now, decode the selected features in X_selection
    for col in categorical_columns:
        if col in X_selection.columns:  # Check if the column is in the selected features
            X_selection.loc[:, col] = encoders[col].inverse_transform(X_selection[col])