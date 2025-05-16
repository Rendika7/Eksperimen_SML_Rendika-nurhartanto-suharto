import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(file_path, save_path):
    """
    Fungsi ini akan memuat dataset, melakukan preprocessing, dan menyimpannya dalam format CSV yang sudah diproses.
    
    :param file_path: Path file dataset raw (CSV).
    :param save_path: Path tempat untuk menyimpan dataset yang sudah diproses.
    """
    # Memuat dataset
    df = pd.read_csv(file_path).drop(columns=['id'])
    
    # 1. Renaming columns dengan nama yang lebih singkat dan lebih bermakna
    df = df.rename(columns={
        'Academic Pressure': 'AcademicsPressure',
        'Work Pressure': 'WorkPressure',
        'Study Satisfaction': 'StudySatisfaction',
        'Job Satisfaction': 'JobSatisfaction',
        'Sleep Duration': 'SleepDuration',
        'Dietary Habits': 'EatingHabits',
        'Have you ever had suicidal thoughts ?': 'SuicidalThoughtsHistory',
        'Work/Study Hours': 'WorkStudyHours',
        'Financial Stress': 'FinancialStress',
        'Family History of Mental Illness': 'MentalIllnessFamilyHistory'
    })
    
    # 2. Dropping Unnecessary Values (Low frequency values)
    threshold = 35  # Threshold to consider for removing values with low frequency
    for kolom in df.columns:
        value_counts = df[kolom].value_counts()
        low_freq_values = value_counts[value_counts < threshold].index
        df = df[~df[kolom].isin(low_freq_values)]

    # 3. Dropping Unnecessary Columns
    df = df.drop(["JobSatisfaction", "WorkPressure", "Profession"], axis=1)

    # 4. Handling Missing Value
    df = df.dropna()  # Drop rows with missing values

    # 5. Feature Engineering - Degree mapping
    degree_mapping = {
        'Class 12': 'Higher Secondary',
        'B.Ed': 'Undergraduate',
        'B.Com': 'Undergraduate',
        'B.Arch': 'Undergraduate',
        'BCA': 'Undergraduate',
        'MSc': 'Postgraduate',
        'B.Tech': 'Undergraduate',
        'MCA': 'Postgraduate',
        'M.Tech': 'Postgraduate',
        'BHM': 'Undergraduate',
        'BSc': 'Undergraduate',
        'M.Ed': 'Postgraduate',
        'B.Pharm': 'Undergraduate',
        'M.Com': 'Postgraduate',
        'BBA': 'Undergraduate',
        'MBBS': 'Postgraduate',
        'LLB': 'Postgraduate',
        'BA': 'Undergraduate',
        'BE': 'Undergraduate',
        'M.Pharm': 'Postgraduate',
        'MD': 'Postgraduate',
        'MBA': 'Postgraduate',
        'MA': 'Postgraduate',
        'PhD': 'Postgraduate',
        'LLM': 'Postgraduate',
        'ME': 'Postgraduate',
        'MHM': 'Postgraduate'
    }
    df["Degree_Category"] = df["Degree"].map(degree_mapping)
    
    df = df.drop("Degree", axis=1)

    # List of categorical columns to encode
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Initialize a dictionary to store the encoders for each column
    encoders = {}

    # Apply encoding and store the encoders
    for col in categorical_columns:
        encoder = LabelEncoder() # Initialize the LabelEncoder
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder  # Store the encoder for each column

    # Then you can proceed with splitting features and target for model training
    X = df.drop(columns=['Depression'])
    y = df['Depression']

    # Fit the model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Extract feature importance
    feature_importance = model.feature_importances_

    # Sorting the feature importance in descending order
    sorted_idx_desc = feature_importance.argsort()[::-1]

    X_selection = X.iloc[:, sorted_idx_desc[:10]]  # Select top 10 important features
    
    cleaned_data = X_selection.copy()
    
    # Now, decode the selected features in X_selection
    for col in categorical_columns:
        if col in cleaned_data.columns:  # Check if the column is in the selected features
            cleaned_data[col] = encoders[col].inverse_transform(cleaned_data[col])

    cleaned_data["Depression"] = y

    # 7. Menyimpan Dataset yang Sudah Diproses
    cleaned_data.to_csv(save_path, index=False)

    return cleaned_data


# Fungsi untuk menjalankan otomatisasi preprocessing
if __name__ == "__main__":
    file_path = 'student-depression-dataset_raw.csv'  # Ganti dengan path dataset raw
    save_path = 'preprocessing/student-depression-dataset_preprocessing.csv'  # Ganti dengan path untuk menyimpan data yang sudah diproses
    
    processed_data = preprocess_data(file_path, save_path)
    print(f"[✅] Dataset berhasil diproses dan disimpan di {save_path}")