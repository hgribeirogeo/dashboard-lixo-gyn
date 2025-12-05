# streamlit_app.py
# Dashboard Detecção de Lixo — Goiânia (SCAN v8 Dark Pro)

import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

st.set_page_config(
    page_title="Dashboard Detecção de Lixo — Goiânia",
    layout="wide",
    page_icon="🗑️",
)

# Tema Dark total (fundo preto, texto claro)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] {
        background-color: #050608;
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] * {
        color: #f5f5f5 !important;
    }
    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #f5f5f5 !important;
    }
    /* Métricas */
    div[data-testid="stMetricValue"] {
        color: #f5f5f5 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #111418 !important;
        border-radius: 10px;
        padding: 10px;
    }
    /* Dataframe */
    .stDataFrame {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Caminhos dos dados (ajuste se necessário)
CSV_DETECCOES = "E:/DADOS/CSVs/DETECCOES_LATLON.csv"
GPKG_BAIRROS = "E:/DADOS/SHP/bairro_gyn.gpkg"
BAIRRO_COL = "name_subdistrict"

# Área mapeada (km²) — valor fixo usado antes
AREA_MAPEADA_KM2 = 641.66

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def plotly_dark_pro_layout():
    """Tema Plotly Black Professional (papel e plot escuros, texto claro)."""
    return dict(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#111418",
        font=dict(color="#f5f5f5", family="Segoe UI, sans-serif"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )


@st.cache_data(show_spinner="Carregando dados...")
def load_data(csv_path: str, bairros_path: str, bairro_col: str):
    # Detecções com lat/lon em WGS84
    df = pd.read_csv(csv_path)

    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter as colunas 'lat' e 'lon' (WGS84).")

    # GeoDataFrame das detecções
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    # Bairros
    gdf_bairros = gpd.read_file(bairros_path)
    if gdf_bairros.crs is None:
        # Se vier sem CRS, assume WGS84 (ajuste se necessário)
        gdf_bairros = gdf_bairros.set_crs(epsg=4326, allow_override=True)
    else:
        gdf_bairros = gdf_bairros.to_crs(epsg=4326)

    # Join espacial (ponto dentro do bairro)
    joined = gpd.sjoin(
        gdf_pts,
        gdf_bairros[[bairro_col, "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.rename(columns={bairro_col: "bairro"})
    joined["bairro"] = joined["bairro"].fillna("Sem bairro")

    # Volta para DataFrame simples (mantendo lat/lon)
    joined["lat"] = joined.geometry.y
    joined["lon"] = joined.geometry.x
    joined = pd.DataFrame(joined.drop(columns="geometry"))

    return joined, gdf_bairros


def criar_mapa(
    df: pd.DataFrame,
    gdf_bairros: gpd.GeoDataFrame,
    mostrar_bairros: bool,
    mostrar_pontos: bool,
    mostrar_heatmap: bool,
    mostrar_clusters: bool,
    fundo_mapa: str,
):
    # Centro do mapa
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    # Tiles
    if fundo_mapa == "dark":
        tiles = "CartoDB dark_matter"
    else:
        tiles = "Esri.WorldImagery"

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=tiles)

    # Bairros
    if mostrar_bairros:
        folium.GeoJson(
            gdf_bairros,
            name="Bairros",
            style_function=lambda x: {
                "color": "#00FFAA",
                "weight": 1,
                "fillOpacity": 0.0,
            },
        ).add_to(m)

    # Heatmap
    if mostrar_heatmap and not df.empty:
        heat_data = df[["lat", "lon"]].values.tolist()
        plugins.HeatMap(
            heat_data,
            radius=18,
            blur=22,
            max_zoom=17,
            min_opacity=0.2,
        ).add_to(m)

    # Pontos / clusters
    if mostrar_pontos and not df.empty:
        if mostrar_clusters:
            cluster = plugins.MarkerCluster(name="Detecções").add_to(m)
            target = cluster
        else:
            target = m

        for _, row in df.iterrows():
            conf = float(row.get("conf", 0))
            # Cor básica pela confiança
            if conf >= 0.85:
                color = "#ffdd00"  # amarelo
            elif conf >= 0.7:
                color = "#00ffff"  # ciano
            else:
                color = "#8888ff"  # roxo claro

            popup = folium.Popup(
                f"Conf: {conf:.2f}<br>Bairro: {row.get('bairro', 'N/A')}<br>IMG: {row.get('img', '')}",
                max_width=250,
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(target)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ============================================================
# CARREGAR DADOS
# ============================================================

try:
    df_all, gdf_bairros = load_data(
        CSV_DETECCOES, GPKG_BAIRROS, BAIRRO_COL
    )
    data_ok = True
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    data_ok = False

# ============================================================
# SIDEBAR — FILTROS
# ============================================================

st.sidebar.title("Filtros e Opções")

conf_min = st.sidebar.slider(
    "Confiança mínima (YOLO)", min_value=0.0, max_value=1.0, value=0.7, step=0.01
)

st.sidebar.markdown("---")

mostrar_bairros = st.sidebar.checkbox("Mostrar bairros", value=True)
mostrar_pontos = st.sidebar.checkbox("Mostrar pontos individuais", value=True)
mostrar_heatmap = st.sidebar.checkbox("Mostrar heatmap", value=True)
mostrar_clusters = st.sidebar.checkbox("Mostrar clusters", value=True)

st.sidebar.markdown("---")
fundo = st.sidebar.radio(
    "Fundo do mapa",
    options=["Dark (padrão)", "Satélite (Esri)"],
    index=0,
)
fundo_mapa = "dark" if fundo.startswith("Dark") else "sat"

# ============================================================
# LAYOUT PRINCIPAL
# ============================================================

st.title("🗑️ Dashboard Detecção de Lixo — Goiânia (SCAN v8)")

if not data_ok:
    st.stop()

# Filtra por confiança
df_filt = df_all[df_all["conf"] >= conf_min].copy()

total_det = len(df_all)
total_filtrado = len(df_filt)

# Ranking por bairro
rank = (
    df_filt.groupby("bairro")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="detecções")
)

# TOP 5
top5 = rank.head(5).copy()
top5_total = top5["detecções"].sum()
concentracao_top5 = (top5_total / total_filtrado * 100) if total_filtrado > 0 else 0.0

# ============================================================
# MÉTRICAS (KPI)
# ============================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Detecções totais", f"{total_det:,}".replace(",", "."))
c2.metric("Detecções filtradas", f"{total_filtrado:,}".replace(",", "."))
c3.metric("Área mapeada (km²)", f"{AREA_MAPEADA_KM2:,.2f}".replace(",", "."))
c4.metric("Concentração TOP 5 bairros", f"{concentracao_top5:,.1f}%")

st.markdown("---")

# ============================================================
# GRÁFICOS TOP 5 — PIZZA + BARRAS CONF
# ============================================================

st.subheader("📊 Indicadores dos TOP 5 Bairros")

if top5.empty or top5["bairro"].eq("Sem bairro").all():
    st.info("Nenhum bairro encontrado (verifique o GPKG e o join espacial).")
else:
    # Pizza — distribuição por bairro no TOP 5
    fig_pie = px.pie(
        top5,
        values="detecções",
        names="bairro",
        hole=0.45,
    )
    fig_pie.update_traces(textinfo="percent+label", textposition="inside")
    fig_pie.update_layout(
        title="Distribuição por bairro (TOP 5)",
        **plotly_dark_pro_layout(),
    )

    # Barras — confiança média por bairro (TOP 5)
    top5_conf = (
        df_filt[df_filt["bairro"].isin(top5["bairro"])]
        .groupby("bairro")["conf"]
        .mean()
        .reset_index()
    )
    fig_bar = px.bar(
        top5_conf,
        x="bairro",
        y="conf",
        color="conf",
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(
        title="Confiança média (YOLO) — TOP 5",
        **plotly_dark_pro_layout(),
    )
    fig_bar.update_yaxes(range=[0, 1], gridcolor="#333333")
    fig_bar.update_xaxes(showgrid=False)

    col_pie, col_bar = st.columns([1, 1.2])
    col_pie.plotly_chart(fig_pie, use_container_width=True)
    col_bar.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ============================================================
# MAPA INTERATIVO
# ============================================================

st.subheader("🗺️ Mapa Interativo")

if df_filt.empty:
    st.warning("Nenhuma detecção com a confiança mínima selecionada.")
else:
    mapa = criar_mapa(
        df_filt,
        gdf_bairros,
        mostrar_bairros=mostrar_bairros,
        mostrar_pontos=mostrar_pontos,
        mostrar_heatmap=mostrar_heatmap,
        mostrar_clusters=mostrar_clusters,
        fundo_mapa=fundo_mapa,
    )
    st_folium(mapa, width=None, height=550)

st.markdown("---")

# ============================================================
# RANKING COMPLETO POR BAIRRO (TOP 20)
# ============================================================

st.subheader("📑 Ranking completo por bairro (TOP 20)")

if rank.empty:
    st.info("Ranking vazio: talvez nenhuma detecção esteja associada a bairros.")
else:
    top20 = rank.head(20).copy()
    top20["% do total filtrado"] = (
        top20["detecções"] / total_filtrado * 100 if total_filtrado > 0 else 0
    )
    top20["% do total filtrado"] = top20["% do total filtrado"].map(
        lambda v: f"{v:,.1f}%"
    )
    st.dataframe(
        top20,
        use_container_width=True,
        hide_index=True,
    )
