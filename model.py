# =====================================================
# MODEL - Ramalan Penggunaan Elektrik Mengikut Sektor
# Sumber: data.gov.my (Suruhanjaya Tenaga Malaysia)
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from data_loader import load_electricity_data

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet tidak tersedia. Guna ARIMA sahaja.")


SECTOR_NAMES = {
    'total': 'Jumlah Keseluruhan',
    'local': 'Penggunaan Tempatan',
    'local_commercial': 'Komersial',
    'local_domestic': 'Kediaman (Domestik)',
    'exports': 'Eksport ke Negara Jiran',
    'losses': 'Kehilangan Elektrik (Losses)'
}


def prepare_sector_data(df, sector):
    """Sediakan data untuk satu sektor dalam format Prophet."""
    sector_df = df[df['sector'] == sector].copy()
    sector_df = sector_df.groupby('date')['consumption'].sum().reset_index()
    sector_df = sector_df.rename(columns={'date': 'ds', 'consumption': 'y'})
    sector_df = sector_df.sort_values('ds').reset_index(drop=True)
    sector_df = sector_df[sector_df['y'] > 0].reset_index(drop=True)
    return sector_df


def train_prophet(train_df, forecast_months=36):
    """Latih model Prophet dan ramal masa depan."""
    if not PROPHET_AVAILABLE:
        return None, None
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    model.fit(train_df)
    
    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)
    
    return model, forecast


def evaluate_model(actual, predicted):
    """Kira metrik ketepatan."""
    try:
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return {'MAPE': round(mape, 2), 'RMSE': round(rmse, 2)}
    except Exception:
        return {'MAPE': 'N/A', 'RMSE': 'N/A'}


def train_all_sectors(df, forecast_months=36):
    """Latih model untuk semua sektor dan simpan hasil."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    results = {}
    all_forecasts = []
    all_history = []
    
    sectors = df['sector'].unique()
    print(f"\n🚀 Melatih model untuk {len(sectors)} sektor...\n")
    for s in sectors:
        print(f"   • {s} → {SECTOR_NAMES.get(s, s)}")
    print()
    
    for i, sector in enumerate(sectors, 1):
        sector_label = SECTOR_NAMES.get(sector, sector)
        print(f"[{i}/{len(sectors)}] Memproses: {sector_label}...")
        sector_df = prepare_sector_data(df, sector)
        
        if len(sector_df) < 12:
            print(f"   ⚠️ Data terlalu sedikit, skip.")
            continue
        
        hist = sector_df.copy()
        hist['sector'] = sector
        hist['sector_label'] = sector_label
        all_history.append(hist)
        
        split = int(len(sector_df) * 0.8)
        train, test = sector_df[:split], sector_df[split:]
        
        if PROPHET_AVAILABLE:
            try:
                _, forecast_val = train_prophet(train, forecast_months=len(test))
                val_pred = forecast_val.tail(len(test))['yhat'].values
                if len(val_pred) == len(test):
                    metrics = evaluate_model(test['y'].values, val_pred)
                else:
                    metrics = {'MAPE': 'N/A', 'RMSE': 'N/A'}
            except Exception as e:
                print(f"   ⚠️ Validasi gagal: {e}")
                metrics = {'MAPE': 'N/A', 'RMSE': 'N/A'}
            
            try:
                model_full, forecast_future = train_prophet(sector_df, forecast_months)
                safe_name = sector.replace('/', '_')
                joblib.dump(model_full, f'models/prophet_{safe_name}.pkl')
                
                future_forecast = forecast_future[
                    forecast_future['ds'] > sector_df['ds'].max()
                ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                future_forecast['sector'] = sector
                future_forecast['sector_label'] = sector_label
                all_forecasts.append(future_forecast)
                
                results[sector_label] = {
                    'model': 'Prophet',
                    'MAPE (%)': metrics['MAPE'],
                    'RMSE': metrics['RMSE']
                }
                print(f"   ✅ Selesai (MAPE: {metrics['MAPE']}%)")
            except Exception as e:
                print(f"   ❌ Latihan gagal: {e}")
        else:
            print("   ⚠️ Prophet tidak tersedia.")
    
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
        print("\n📊 Ringkasan Prestasi Model (MAPE < 10% = sangat baik):")
        print(summary)
    
    print("\n🎉 Selesai! Sistem sedia untuk dashboard.")
    return all_forecasts, results


if __name__ == "__main__":
    df = load_electricity_data()
    forecasts, summary = train_all_sectors(df, forecast_months=36)