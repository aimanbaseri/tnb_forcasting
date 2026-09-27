# =====================================================
# TNB LOAD FORECASTING DASHBOARD
# Ramalan Penggunaan Elektrik Malaysia (2025-2027)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_loader import load_electricity_data

st.set_page_config(
    page_title="TNB Load Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

SECTOR_NAMES = {
    'total': 'Jumlah Keseluruhan',
    'local': 'Penggunaan Tempatan',
    'local_commercial': 'Komersial',
    'local_domestic': 'Kediaman (Domestik)',
    'exports': 'Eksport',
    'losses': 'Kehilangan Elektrik (Losses)'
}

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #003d7a 0%, #0066cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ TNB Load Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Ramalan Penggunaan Elektrik Malaysia | 2018 - 2027</div>', unsafe_allow_html=True)


@st.cache_data
def get_data():
    df = load_electricity_data()
    df['date'] = pd.to_datetime(df['date'])
    df['sector_label'] = df['sector'].map(SECTOR_NAMES).fillna(df['sector'])
    return df

@st.cache_data
def get_forecasts():
    if os.path.exists('outputs/forecast_2025_2027.csv'):
        df = pd.read_csv('outputs/forecast_2025_2027.csv', parse_dates=['ds'])
        return df
    return None

@st.cache_data
def get_performance():
    if os.path.exists('outputs/model_performance.csv'):
        return pd.read_csv('outputs/model_performance.csv', index_col=0)
    return None


# ==========================================
# AUTO-GENERATE FORECAST (untuk Streamlit Cloud)
# ==========================================
if not os.path.exists('outputs/forecast_2025_2027.csv'):
    with st.spinner("🔮 Melatih model buat kali pertama... (2-3 minit). Sila tunggu."):
        try:
            from model import train_all_sectors
            df_raw = load_electricity_data()
            train_all_sectors(df_raw, forecast_months=36)
            st.success("✅ Model selesai dilatih!")
        except Exception as e:
            st.error(f"❌ Gagal melatih model: {e}")

data = get_data()
forecasts = get_forecasts()
performance = get_performance()


with st.sidebar:
    st.header("🎛️ Penapis")
    
    sectors = sorted(data['sector'].unique())
    sector_labels = {s: SECTOR_NAMES.get(s, s) for s in sectors}
    
    selected_label = st.selectbox(
        "Pilih Sektor:",
        ["SEMUA SEKTOR"] + list(sector_labels.values())
    )
    
    if selected_label != "SEMUA SEKTOR":
        selected_sector = [k for k, v in sector_labels.items() if v == selected_label][0]
    else:
        selected_sector = None
    
    st.markdown("---")
    st.header("ℹ️ Tentang Sistem")
    st.markdown("""
    <div class="source-box">
    <b>Tujuan:</b> Meramal permintaan elektrik Malaysia 
    untuk membantu perancangan kapasiti grid TNB.<br><br>
    <b>Model:</b> ARIMA (Time Series)<br>
    <b>Sumber:</b> data.gov.my<br>
    <b>Lesen:</b> CC BY 4.0
    </div>
    """, unsafe_allow_html=True)


if selected_sector:
    kpi_data = data[data['sector'] == selected_sector]
else:
    kpi_data = data[data['sector'] == 'total']

total_consumption = kpi_data['consumption'].sum()
avg_monthly = kpi_data['consumption'].mean()
peak = kpi_data['consumption'].max()

y2018 = kpi_data[kpi_data['date'].dt.year == 2018]['consumption'].sum()
y2023 = kpi_data[kpi_data['date'].dt.year == 2023]['consumption'].sum()
growth = ((y2023 / y2018) - 1) * 100 if y2018 > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("⚡ Jumlah Penggunaan", f"{total_consumption:,.0f} GWh")
with col2:
    st.metric("📊 Purata Bulanan", f"{avg_monthly:,.0f} GWh")
with col3:
    st.metric("🔥 Peak (Tertinggi)", f"{peak:,.0f} GWh")
with col4:
    st.metric("📈 Pertumbuhan 2018→2023", f"+{growth:.1f}%")

st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Analisis Sejarah",
    "🔮 Ramalan 2025-2027",
    "🏆 Prestasi Model",
    "📋 Data"
])


# ==========================================
# TAB 1: ANALISIS SEJARAH
# ==========================================
with tab1:
    st.subheader("📈 Analisis Penggunaan Elektrik (2018-2024)")
    
    if selected_sector:
        trend = data[data['sector'] == selected_sector].sort_values('date')
        title = f"Trend Penggunaan - {selected_label}"
    else:
        trend = data[data['sector'] == 'total'].sort_values('date')
        title = "Trend Penggunaan Elektrik Malaysia (Jumlah Keseluruhan)"
    
    fig = px.line(trend, x='date', y='consumption',
                  title=title,
                  labels={'consumption': 'Penggunaan (GWh)', 'date': 'Tarikh'},
                  color_discrete_sequence=['#0066cc'])
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Perbandingan Sektor (2018-2024)**")
        sector_totals = data.groupby('sector_label')['consumption'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(sector_totals, x='consumption', y='sector_label', orientation='h',
                     color='consumption', color_continuous_scale='Blues',
                     labels={'consumption': 'Jumlah (GWh)', 'sector_label': ''})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.write("**Corak Bulanan (Seasonality)**")
        data['month'] = data['date'].dt.month
        if selected_sector:
            monthly = data[data['sector'] == selected_sector].groupby('month')['consumption'].mean().reset_index()
        else:
            monthly = data[data['sector'] == 'total'].groupby('month')['consumption'].mean().reset_index()
        fig = px.line(monthly, x='month', y='consumption', markers=True,
                      labels={'consumption': 'Purata (GWh)', 'month': 'Bulan'},
                      color_discrete_sequence=['#e74c3c'])
        fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 2: RAMALAN
# ==========================================
with tab2:
    st.subheader("🔮 Ramalan Penggunaan Elektrik (2025-2027)")
    
    if forecasts is None:
        st.warning("⚠️ Model sedang dilatih. Sila refresh halaman ini selepas 2-3 minit.")
    else:
        if selected_sector:
            fc = forecasts[forecasts['sector'] == selected_sector].sort_values('ds')
        else:
            fc = forecasts[forecasts['sector'] == 'total'].sort_values('ds')
        
        if selected_sector:
            hist = data[data['sector'] == selected_sector][['date', 'consumption']].rename(columns={'date': 'ds', 'consumption': 'value'})
        else:
            hist = data[data['sector'] == 'total'][['date', 'consumption']].rename(columns={'date': 'ds', 'consumption': 'value'})
        
        hist['type'] = 'Sejarah'
        fc_plot = fc[['ds', 'yhat']].rename(columns={'yhat': 'value'})
        fc_plot['type'] = 'Ramalan'
        
        combined = pd.concat([hist, fc_plot], ignore_index=True).sort_values('ds')
        
        fig = px.line(combined, x='ds', y='value', color='type',
                      title=f'Ramalan - {selected_label}',
                      labels={'value': 'Penggunaan (GWh)', 'ds': 'Tarikh'},
                      color_discrete_map={'Sejarah': '#0066cc', 'Ramalan': '#e74c3c'})
        
        # Tukar tarikh ke Unix timestamp (untuk elak bug Plotly)
        last_date_ms = int(hist['ds'].max().timestamp() * 1000)
        last_forecast_ms = int(fc['ds'].max().timestamp() * 1000)
        
        # Zon berlorek untuk bahagian ramalan
        fig.add_vrect(
            x0=last_date_ms,
            x1=last_forecast_ms,
            fillcolor="red",
            opacity=0.08,
            layer="below",
            line_width=0,
            annotation_text="🔮 Zon Ramalan",
            annotation_position="top left",
            annotation=dict(font_size=11, font_color="#e74c3c")
        )
        
        # Garis pemisah
        fig.add_vline(
            x=last_date_ms,
            line_dash="dash",
            line_color="#e74c3c",
            line_width=1.5
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Ringkasan Ramalan Tahunan")
        col_a, col_b, col_c = st.columns(3)
        
        for col, year in zip([col_a, col_b, col_c], [2025, 2026, 2027]):
            year_total = fc[fc['ds'].dt.year == year]['yhat'].sum()
            with col:
                st.metric(f"Ramalan {year}", f"{year_total:,.0f} GWh")


# ==========================================
# TAB 3: PRESTASI MODEL
# ==========================================
with tab3:
    st.subheader("🏆 Prestasi Model Ramalan")
    
    if performance is None:
        st.warning("⚠️ Model sedang dilatih. Sila refresh halaman ini selepas 2-3 minit.")
    else:
        st.write("**Metrik Ketepatan Setiap Sektor**")
        st.dataframe(performance, use_container_width=True)
        
        mape_values = pd.to_numeric(performance['MAPE (%)'], errors='coerce').dropna()
        if len(mape_values) > 0:
            avg_mape = mape_values.mean()
            st.metric("Purata MAPE (semua sektor)", f"{avg_mape:.2f}%",
                      help="MAPE < 10% = sangat bagus | 10-20% = boleh diterima | >20% = perlu perbaiki")
            
            fig = px.bar(performance.reset_index(), x='index', y='MAPE (%)',
                         color='MAPE (%)', color_continuous_scale='RdYlGn_r',
                         labels={'index': 'Sektor', 'MAPE (%)': 'MAPE (%)'})
            fig.add_hline(y=10, line_dash="dash", line_color="green",
                          annotation_text="Target: 10%")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 4: DATA
# ==========================================
with tab4:
    st.subheader("📋 Data Mentah")
    
    if selected_sector:
        display_data = data[data['sector'] == selected_sector].sort_values('date', ascending=False)
    else:
        display_data = data.sort_values('date', ascending=False)
    
    st.write(f"Menunjukkan **{len(display_data)}** baris data.")
    st.dataframe(display_data[['date', 'sector_label', 'consumption']], use_container_width=True)
    
    csv = display_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Muat Turun Data (CSV)",
        csv,
        "tnb_electricity_data.csv",
        "text/csv"
    )


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    ⚡ TNB Load Forecasting System | Python + ARIMA + Streamlit + Plotly<br>
    Sumber Data: <b>data.gov.my</b> (Suruhanjaya Tenaga Malaysia) | Lesen: CC BY 4.0
</div>
""", unsafe_allow_html=True)
