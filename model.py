# =====================================================
# MODEL - Ramalan Penggunaan Elektrik (ARIMA)
# Sumber: data.gov.my (Suruhanjaya Tenaga Malaysia)
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from data_loader import load_electricity_data


SECTOR_NAMES = {
    'total': 'Jumlah Keseluruhan',
    'local': 'Penggunaan Tempatan',
    'local_commercial': 'Komersial',
    'local_domestic': 'Kediaman (Domestik)',
    'exports': 'Eksport ke Negara Jiran',
    'losses': 'Kehilangan Elektrik (Losses)'
}


def prepare_sector_data(df, sector):
    """Sediakan data untuk satu sektor."""
    sector_df = df[df['sector'] == sector].copy()
    sector_df = sector_df.groupby('date')['consumption'].sum().reset_index()
    sector_df = sector_df.rename(columns={'date': 'ds', 'consumption': 'y'})
    sector_df = sector_df.sort_values('ds').reset_index(drop=True)
    sector_df = sector_df[sector_df['y'] > 0].reset_index(drop=True)
    return sector_df


def train_arima(train_df, forecast_months=36):
    """Latih model ARIMA dan ramal masa depan."""
    try:
        model = ARIMA(train_df['y'], order=(2, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_months)
        
        last_date = train_df['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_months, freq='MS'
        )
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast.values,
            'yhat_lower': forecast.values * 0.95,
            'yhat_upper': forecast.values * 1.05
        })
        return fitted, forecast_df
    except Exception as e:
        print(f"ARIMA gagal: {e}")
        return None, None


def evaluate_model(actual, predicted):
    """Kira metrik ketepatan."""
    try:
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return {'MAPE': round(mape, 2), 'RMSE': round(rmse, 2)}
    except Exception:
        return {'MAPE': 'N/A', 'RMSE': 'N/A'}


def train_all_sectors(df, forecast_months=36):
    """Latih model ARIMA untuk semua sektor."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    results = {}
    all_forecasts = []
    all_history = []
    
    sectors = df['sector'].unique()
    print(f"\n🚀 Melatih model ARIMA untuk {len(sectors)} sektor...\n")
    
    for i, sector in enumerate(sectors, 1):
        sector_label = SECTOR_NAMES.get(sector, sector)
        print(f"[{i}/{len(sectors)}] Memproses: {sector_label}...")
        sector_df = prepare_sector_data(df, sector)
        
        if len(sector_df) < 24:
            print(f"   ⚠️ Data terlalu sedikit, skip.")
            continue
        
        # Simpan sejarah
        hist = sector_df.copy()
        hist['sector'] = sector
        hist['sector_label'] = sector_label
        all_history.append(hist)
        
        # Split 80/20 untuk validasi
        split = int(len(sector_df) * 0.8)
        train, test = sector_df[:split], sector_df[split:]
        
        # Validasi
        _, forecast_val = train_arima(train, forecast_months=len(test))
        if forecast_val is not None and len(forecast_val) == len(test):
            metrics = evaluate_model(test['y'].values, forecast_val['yhat'].values)
        else:
            metrics = {'MAPE': 'N/A', 'RMSE': 'N/A'}
        
        # Latih dengan data penuh
        model_full, forecast_future = train_arima(sector_df, forecast_months)
        if model_full is None:
            print(f"   ❌ Gagal latih {sector}")
            continue
        
        # Simpan model
        safe_name = sector.replace('/', '_')
        joblib.dump(model_full, f'models/arima_{safe_name}.pkl')
        
        forecast_future['sector'] = sector
        forecast_future['sector_label'] = sector_label
        all_forecasts.append(forecast_future)
        
        results[sector_label] = {
            'model': 'ARIMA',
            'MAPE (%)': metrics['MAPE'],
            'RMSE': metrics['RMSE']
        }
        print(f"   ✅ Selesai (MAPE: {metrics['MAPE']}%)")
    
    # Simpan output
    if all_forecasts:
        final_forecasts = pd.concat(all_forecasts, ignore_index=True)
        final_forecasts.to_csv('outputs/forecast_2025_2027.csv', index=False)
        print(f"\n✅ Ramalan disimpan: outputs/forecast_2025_2027.csv ({len(final_forecasts)} baris)")
    
    if all_history:
        final_history = pd.concat(all_history, ignore_index=True)
        final_history.to_csv('outputs/history_data.csv', index=False)
    
    if results:
        summary = pd.DataFrame(results).T
        summary.to_csv('outputs/model_performance.csv')
        print("\n📊 Ringkasan Prestasi Model:")
        print(summary)
    
    print("\n🎉 Selesai!")
    return all_forecasts, results


if __name__ == "__main__":
    df = load_electricity_data()
    forecasts, summary = train_all_sectors(df, forecast_months=36)
