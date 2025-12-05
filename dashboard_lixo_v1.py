# ============================================================
# Dashboard Detecção de Lixo — Goiânia (SCAN v6)
# Versão FINAL — v5
# - Topo branco removido
# - Layout 100% dark
# - Heatmap corrigido
# - Satélite opcional
# - Legenda estilizada
# ============================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
from pyproj import Transformer
import rasterio
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_folium import st_folium

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard Detecção de Lixo — Goiânia (SCAN v6)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS GLOBAL — DARK SUAVE + HEADER
# ============================================================
st.markdown("""
<style>

/* fundo geral */
[data-testid="stAppViewContainer"] {
    background-color: #020617 !important;
}

/* HEADER (remover topo branco) */
[data-testid="stHeader"] {
    background-color: #020617 !important;
    border-bottom: 1px solid #111827;
}

/* barra superior invisível */
header[data-testid="stToolbar"] {
    display: none !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617 !important;
    border-right: 1px solid #111827 !important;
}

/* títulos */
h1, h2, h3, h4, h5, h6 {
    color: #e5e7eb !important;
}

/* textos */
p, span, label, div, li {
    color: #e5e7eb !important;
}

/* KPIs */
.metric-container {
    background-color: #020617;
    padding: 0.75rem;
    border-radius: 10px;
    border: 1px solid #1f2937;
    box-shadow: 0 0 12px rgba(15,23,42,0.5);
}
.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
}
.metric-value {
    color: #e5e7eb;
    font-size: 1.4rem;
    font-weight: bold;
}

/* área do app */
.block-container {
    padding-top: 1rem !important;
}

/* scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #020617;
}
::-webkit-scrollbar-thumb {
    background-color: #4b5563;
}

/* remover fundo branco dos gráficos */
.stPlotlyChart, .stChart, .stImage {
    background-color: #020617 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ARQUIVOS
# ============================================================
CSV_DETECCOES = r"E:/PROJETOS/lixo_vc/scan_full_v6_1_no_filter_C/detecoes_scan_full.csv"
BAIRROS_GPKG = r"E:/DADOS/SHP/bairro_gyn.gpkg"
BAIRRO_COL = "name_subdistrict"
MOSAICO_PATH = r"E:/DADOS/ORTOFOTO_GOIANIA_2024/mosaico_2024.cog"

CRS_TILE = "EPSG:31982"
CRS_WGS = "EPSG:4326"
LIXO_CLASS_ID = 1

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Filtros e Opções")

conf_min = st.sidebar.slider("Confiança mínima (YOLO)", 0.5, 0.99, 0.70, 0.01)
mostrar_bairros = st.sidebar.checkbox("Mostrar bairros", True)
mostrar_pontos = st.sidebar.checkbox("Mostrar pontos individuais", True)
mostrar_heatmap = st.sidebar.checkbox("Mostrar heatmap", True)
mostrar_clusters = st.sidebar.checkbox("Mostrar clusters", True)

fundo_mapa = st.sidebar.radio(
    "Fundo do mapa",
    ["Dark (padrão)", "Satélite (Google)"],
    index=0
)

coluna_bairro = st.sidebar.text_input("Coluna de bairro no GPKG", BAIRRO_COL)

# ============================================================
# FUNÇÃO DE CARREGAMENTO
# ============================================================
@st.cache_data
def carregar_dados(csv_path, gpkg_path, coluna_bairro):
    df = pd.read_csv(csv_path)
    df = df[df["class"] == LIXO_CLASS_ID].copy()

    src = rasterio.open(MOSAICO_PATH)
    transform = src.transform

    # centro dos bounding boxes
    cx = (df["x1"] + df["x2"]) / 2
    cy = (df["y1"] + df["y2"]) / 2
    px = df["col"] + cx
    py = df["row"] + cy

    xs, ys = rasterio.transform.xy(transform, py, px)
    df["x_31982"] = xs
    df["y_31982"] = ys

    transformer = Transformer.from_crs(CRS_TILE, CRS_WGS, always_xy=True)
    lon, lat = transformer.transform(xs, ys)
    df["lon"] = lon
    df["lat"] = lat

    bairros = gpd.read_file(gpkg_path).to_crs(CRS_TILE)

    gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x_31982, df.y_31982), crs=CRS_TILE)

    join = gpd.sjoin(
        gdf_pts,
        bairros[[coluna_bairro, "geometry"]],
        how="left",
        predicate="within"
    ).rename(columns={coluna_bairro: "bairro"})

    # área total
    try:
        area_km2 = bairros.geometry.union_all().area / 1e6
    except:
        area_km2 = bairros.geometry.unary_union.area / 1e6

    return join, bairros.to_crs(CRS_WGS), area_km2

# ============================================================
# PROCESSAMENTO
# ============================================================
gdf_det, gdf_bairros_wgs, area_total_km2 = carregar_dados(
    CSV_DETECCOES, BAIRROS_GPKG, coluna_bairro
)
gdf_filtrado = gdf_det[gdf_det["conf"] >= conf_min]

rank = gdf_filtrado.dropna(subset=["bairro"]).groupby("bairro").size().sort_values(ascending=False)
top5 = rank.head(5)
perc_top5 = (top5.sum() / max(len(gdf_filtrado), 1)) * 100

# ============================================================
# TÍTULO & KPIs
# ============================================================
st.markdown("<h2>📊 Dashboard Detecção de Lixo — Goiânia (SCAN v6)</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Detecções totais</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{len(gdf_det):,}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Detecções filtradas</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{len(gdf_filtrado):,}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Área mapeada (km²)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{area_total_km2:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Concentração TOP 5 bairros</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{perc_top5:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MAPA + GRÁFICOS
# ============================================================
st.markdown("---")

col_map, col_plot = st.columns([3.6, 2])

with col_map:

    st.markdown("<h4>🗺️ Mapa Interativo</h4>", unsafe_allow_html=True)

    center = [
        gdf_filtrado["lat"].median() if len(gdf_filtrado) > 0 else -16.6869,
        gdf_filtrado["lon"].median() if len(gdf_filtrado) > 0 else -49.2648
    ]

    if fundo_mapa == "Dark (padrão)":
        tileset = "CartoDB dark_matter"
        tiles_attr = "CartoDB"
    else:
        tileset = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
        tiles_attr = "Google Satellite"

    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles=tileset,
        attr=tiles_attr
    )

    # bairros
    if mostrar_bairros:
        folium.GeoJson(
            gdf_bairros_wgs.to_json(),
            style_function=lambda x: {"color": "#22c55e55", "weight": 1}
        ).add_to(m)

    # pontos
    if mostrar_pontos:
        for _, r in gdf_filtrado.iterrows():
            folium.CircleMarker(
                [r.lat, r.lon], radius=2, color="#38bdf8", fill=True, fill_opacity=0.8
            ).add_to(m)

    # clusters
    if mostrar_clusters:
        mc = MarkerCluster()
        for _, r in gdf_filtrado.iterrows():
            folium.CircleMarker(
                [r.lat, r.lon], radius=3, color="#f97316", fill=True, fill_opacity=0.9
            ).add_to(mc)
        mc.add_to(m)

    # heatmap
    if mostrar_heatmap:
        HeatMap(
            gdf_filtrado[["lat", "lon"]].values,
            radius=35,
            blur=45,
            gradient={0: "blue", 0.5: "yellow", 1: "red"}
        ).add_to(m)

    st_folium(m, height=650, width=None)

    # legenda
    st.markdown("""
        <div style="background:#0f172a; padding:10px; width:350px; border-radius:8px; border:1px solid #334155;">
            <b>🔥 Legenda do Heatmap</b><br>
            🔵 Baixa intensidade<br>
            🟡 Média intensidade<br>
            🔴 Alta intensidade
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# GRÁFICOS TOP 5
# ============================================================
with col_plot:

    st.markdown("<h4>📊 Análises por Bairro</h4>", unsafe_allow_html=True)

    if not top5.empty:

        # barplot
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#020617")
        ax.set_facecolor("#020617")

        ax.barh(top5.index, top5.values, color="#22c55e")
        ax.invert_yaxis()
        ax.set_title("TOP 5 bairros com mais detecções", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#4b5563")
        st.pyplot(fig)

        # piechart
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        fig2.patch.set_facecolor("#020617")
        ax2.set_facecolor("#020617")

        ax2.pie(
            top5.values, labels=top5.index,
            autopct="%1.1f%%",
            textprops={"color": "white"},
            startangle=140
        )
        ax2.set_title("Participação dos TOP 5", color="white")
        st.pyplot(fig2)

# ============================================================
# RANKING COMPLETO
# ============================================================
st.markdown("---")
st.subheader("📋 Ranking completo por bairro")

if not rank.empty:
    df_rank = rank.reset_index()
    df_rank.columns = ["bairro", "n_lixo"]
    df_rank["rank"] = df_rank["n_lixo"].rank(ascending=False).astype(int)
    df_rank = df_rank.sort_values("rank")

    st.dataframe(df_rank)
