# =====================================================
# DATA LOADER - Sumber Data Elektrik Malaysia
# =====================================================
# Sumber: data.gov.my (Suruhanjaya Tenaga / Energy Commission)
# Lesen: CC BY 4.0
# =====================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_electricity_data():
    """
    Memuatkan data penggunaan elektrik Malaysia.
    Jika API gagal, ia akan jana data sintetik yang realistik.
    """
    try:
        # Cuba ambil data dari data.gov.my
        # Endpoint sebenar mungkin berbeza - semak di data.gov.my/data-catalogue
        url = "https://storage.data.gov.my/energy/electricity_consumption.parquet"
        df = pd.read_parquet(url)
        print("✅ Data dari data.gov.my berjaya dimuatkan.")
        return df
    except Exception as e:
        print(f"⚠️  Gagal ambil dari API: {e}")
        print("📊 Menjana data sintetik yang realistik untuk demo...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """
    Menjana data penggunaan elektrik bulanan Malaysia (2018-2024)
    berdasarkan trend sebenar: peningkatan ~3% setahun + seasonal pattern.
    """
    np.random.seed(42)
    
    # Negeri-negeri di Malaysia
    states = [
        "Selangor", "Johor", "Pulau Pinang", "Perak", "Sarawak", "Sabah",
        "Kuala Lumpur", "Kedah", "Pahang", "Negeri Sembilan", 
        "Melaka", "Terengganu", "Kelantan", "Perlis", "Labuan", "Putrajaya"
    ]
    
    # Baseline permintaan setiap negeri (GWh/month) - anggaran realistik
    baseline = {
        "Selangor": 3500, "Johor": 2200, "Pulau Pinang": 1800, "Perak": 1500,
        "Sarawak": 1300, "Sabah": 1100, "Kuala Lumpur": 2000, "Kedah": 900,
        "Pahang": 800, "Negeri Sembilan": 700, "Melaka": 600, "Terengganu": 550,
        "Kelantan": 500, "Perlis": 200, "Labuan": 150, "Putrajaya": 400
    }
    
    rows = []
    start_date = datetime(2018, 1, 1)
    
    for state in states:
        base = baseline[state]
        for month_offset in range(84):  # 7 tahun = 84 bulan
            date = start_date + timedelta(days=30 * month_offset)
            
            # Trend: kenaikan ~3% setahun
            trend = 1 + (0.03 / 12) * month_offset
            
            # Seasonality: lebih tinggi pada bulan panas (Mac-Sept)
            season = 1 + 0.15 * np.sin(2 * np.pi * (date.month - 3) / 12)
            
            # Noise rawak
            noise = np.random.normal(1, 0.05)
            
            # Kira nilai
            demand = base * trend * season * noise
            
            rows.append({
                "date": date.replace(day=1),
                "state": state,
                "demand_gwh": round(demand, 2)
            })
    
    df = pd.DataFrame(rows)
    print(f"✅ Data sintetik dijana: {len(df)} baris")
    return df


if __name__ == "__main__":
    data = load_electricity_data()
    print(data.head())
    print(f"\nBentuk data: {data.shape}")
    print(f"Julat tarikh: {data['date'].min()} hingga {data['date'].max()}")