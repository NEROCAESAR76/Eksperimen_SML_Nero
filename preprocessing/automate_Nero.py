import pandas as pd
import numpy as np
import os

def run_preprocessing(input_path, output_path):
    print("=== Memulai Proses Preprocessing Otomatis ===")
    
    # 1. Load Data Mentah
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Waduh, file mentah gak ketemu di {input_path}")
        
    df = pd.read_csv(input_path)
    
    # 2. Buang kolom yang gak kepake buat Machine Learning
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # 3. Beresin nilai kosong di TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # 4. Ubah teks jadi angka (Encoding)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
        
    # 5. Simpan Data Bersih
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"=== Mantap! Data bersih udah disimpen di: {output_path} ===")

if __name__ == "__main__":
    # Path ini dibikin relative dari folder utama (root) biar jalan di GitHub
    INPUT = "namadataset_raw/telco_churn.csv"
    OUTPUT = "preprocessing/namadataset_preprocessing/telco_churn_clean.csv"
    run_preprocessing(INPUT, OUTPUT)