import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """
    Fungsi untuk membaca data dari file CSV
    """
    data = pd.read_csv(path)
    print(f"Data berhasil diload: {data.shape}")
    return data

def preprocess_data(df):
    """
    Fungsi utama untuk membersihkan data (Preprocessing)
    Sesuai eksperimen di notebook.
    """
    # 1. Drop kolom yang tidak berguna untuk model
    print("Menghapus kolom tidak relevan...")
    cols_to_drop = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID', 'PreviousTransactionDate']
    # Cek dulu apakah kolom ada biar gak error kalau dirun 2x
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    
    # 2. Handling Date: Ubah string ke datetime & Ambil Jam
    if 'TransactionDate' in df.columns:
        print("Memproses kolom tanggal...")
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['TransactionHour'] = df['TransactionDate'].dt.hour
        df = df.drop(columns=['TransactionDate'])
    
    # 3. Encoding: Mengubah kategori jadi angka
    # Kita pakai LabelEncoder sederhana
    print("Melakukan Encoding...")
    cat_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation', 'LoginAttempts']
    
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # 4. Scaling: Menyamakan skala angka
    print("Melakukan Scaling...")
    num_cols = ['TransactionAmount', 'AccountBalance', 'TransactionDuration', 'CustomerAge']
    # Pastikan kolom ada
    actual_num_cols = [c for c in num_cols if c in df.columns]
    
    scaler = StandardScaler()
    df[actual_num_cols] = scaler.fit_transform(df[actual_num_cols])
    
    return df

if __name__ == "__main__":
    # --- KONFIGURASI FILE ---
    # Sesuaikan path ini dengan lokasi file lo nanti di GitHub/Lokal
    input_file = "../bank_transactions_data_2.csv"  # Asumsi file csv ada di folder luar 'preprocessing'
    output_file = "namadataset_preprocessing/bank_data_preprocessing.csv" 
    
    # --- EKSEKUSI ---
    try:
        # 1. Load
        raw_df = load_data(input_file)
        
        # 2. Preprocess
        clean_df = preprocess_data(raw_df)
        
        # 3. Save
        clean_df.to_csv(output_file, index=False)
        print(f"Sukses! Data bersih tersimpan di: {output_file}")
        
    except FileNotFoundError:
        print("Error: File input tidak ditemukan. Cek path 'input_file' di script.")
    except Exception as e:
        print(f"Terjadi Error: {e}")