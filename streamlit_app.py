# ============================================================
# DASHBOARD LIXO GOIÂNIA — ULTRA-OTIMIZADO (v5.2 Final)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster, HeatMap
from streamlit_folium import st_folium
import plotly.express as px

# ================================
# CONFIG TEMA DARK
# ================================
st.set_page_config(
    page_title="Detecção de Lixo Urbano - Goiânia (SCAN v7)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body { background-color: #0e0e0e; color: #f0f0f0; }
    .stApp { background-color: #0e0e0e; color: #f0f0f0; }
    .css-1d391kg, .css-18e3th9 { background-color: #111111 !important; }
    .stMetric { 
        background-color: #222222; 
        padding: 8px; 
        border-radius: 10px; 
    }
    .stMetric label, .stMetric [data-testid="stMetricValue"] { 
        color: #ffffff !important; 
    }
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 1) CARREGAMENTO DO CSV (CORRIGIDO: SEM UI DENTRO DO CACHE)
# ============================================================

@st.cache_data(ttl=3600)
def load_and_process_csv(path):
    """
    Carrega CSV, limpa dados e retorna o DataFrame + flag se houve troca de colunas.
    Retorna: (pd.DataFrame, bool)
    """
    # 1. Carregar
    try:
        df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return None, False

    df.columns = df.columns.str.lower()
    
    # 2. Parse vetorizado (rápido)
    def parse_coord_vetorizado(series):
        # Remove espaços e troca vírgula por ponto
        s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    if "lat" in df.columns:
        df["lat_clean"] = parse_coord_vetorizado(df["lat"])
    if "lon" in df.columns:
        df["lon_clean"] = parse_coord_vetorizado(df["lon"])
        
    # Remove linhas onde conversão falhou
    df = df.dropna(subset=["lat_clean", "lon_clean"])

    # 3. DETECÇÃO INTELIGENTE DE TROCA DE COLUNAS
    # Goiânia: Lat ~ -16.6, Lon ~ -49.2
    # Se a média da 'lat' for < -40 (lon) e 'lon' for > -20 (lat), estão trocadas.
    media_lat_orig = df["lat_clean"].mean()
    media_lon_orig = df["lon_clean"].mean()

    swapped = False
    
    if (media_lat_orig < -40) and (media_lon_orig > -20):
        swapped = True
        df["lat"] = df["lon_clean"]
        df["lon"] = df["lat_clean"]
    else:
        df["lat"] = df["lat_clean"]
        df["lon"] = df["lon_clean"]

    # 4. Filtro final de sanidade geográfica (Bounding Box estendido de GO)
    df = df[
        (df["lat"] >= -20) & (df["lat"] <= -12) & 
        (df["lon"] >= -53) & (df["lon"] <= -45)
    ]
    
    # Limpeza final
    if "bairro" not in df.columns:
        df["bairro"] = "Não identificado"
    else:
        df["bairro"] = df["bairro"].fillna("Não identificado")
    
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
    df = df.dropna(subset=["conf"])
    
    # Remove colunas auxiliares para economizar memória
    df = df.drop(columns=["lat_clean", "lon_clean"])
    
    return df, swapped

# ============================================================
# EXECUÇÃO DO CARREGAMENTO (COM TRATAMENTO DE RETORNO)
# ============================================================

CSV_PATH = "data/DETECCOES_LATLON4.csv"

try:
    # A função agora retorna DOIS valores
    resultado = load_and_process_csv(CSV_PATH)
    
    if resultado[0] is None:
        st.error(f"Arquivo não encontrado: {CSV_PATH}")
        st.stop()
        
    df, foi_corrigido = resultado
    
    # Exibe o aviso FORA da função cacheada (Isso corrige o erro do Streamlit)
    if foi_corrigido:
        st.toast("Aviso: Coordenadas Lat/Lon invertidas foram corrigidas automaticamente.", icon="🔧")
        
except Exception as e:
    st.error(f"Erro crítico ao processar dados: {e}")
    st.stop()


# ============================================================
# 2) SIDEBAR — FILTROS
# ============================================================

st.sidebar.header("Filtros")

# Lista de bairros
if not df.empty:
    lista_bairros = ["Todos"] + sorted(df["bairro"].unique().tolist())
else:
    lista_bairros = ["Todos"]

bairro_sel = st.sidebar.selectbox("Bairro:", lista_bairros)

conf_min = st.sidebar.slider("Confiança mínima (YOLO)", 0.0, 1.0, 0.50, 0.01)

# Controle de performance
max_markers = st.sidebar.slider(
    "Máx. pontos no mapa", 
    100, 5000, 1000, 100,
    help="Reduza se o mapa estiver lento."
)

tipo_mapa = st.sidebar.radio("Visualização:", ["Marcadores", "Mapa de Calor"])

estilo_mapa = st.sidebar.selectbox(
    "Estilo do mapa:",
    ["CartoDB Dark", "Google Streets", "OpenStreetMap", "Satellite"]
)

# Aplicar filtros
df_filt = df[df["conf"] >= conf_min]

if bairro_sel != "Todos":
    df_filt = df_filt[df_filt["bairro"] == bairro_sel]


# ============================================================
# 3) MÉTRICAS
# ============================================================

col1, col2, col3 = st.columns(3)
col1.metric("Detecções totais", len(df))
col2.metric("Detecções filtradas", len(df_filt))
col3.metric("Bairros encontrados", df_filt["bairro"].nunique())


# ============================================================
# 4) LAYOUT: MAPA + GRÁFICO
# ============================================================

st.subheader("🗺️ Visualização Geográfica e Distribuição")

col_mapa, col_pizza = st.columns([2, 1])

with col_mapa:
    if len(df_filt) > 0:
        # Amostragem para performance se exceder limite
        df_map = df_filt
        
        # Aviso de amostragem (apenas se for marcadores e exceder limite)
        if tipo_mapa == "Marcadores" and len(df_filt) > max_markers:
            st.info(f"⚡ Exibindo amostra de {max_markers} pontos (Total filtrado: {len(df_filt)})")
            df_map = df_filt.sample(n=max_markers, random_state=42)
        
        # Centro dinâmico
        center_lat = df_map["lat"].mean()
        center_lon = df_map["lon"].mean()
        
        # Configuração de Tiles
        tiles_cfg = {
            "Google Streets": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "CartoDB Dark": "CartoDB DarkMatter",
            "OpenStreetMap": "OpenStreetMap"
        }
        
        attr_dict = {
            "Google Streets": "Google",
            "Satellite": "Esri",
            "CartoDB Dark": "CartoDB",
            "OpenStreetMap": "OpenStreetMap"
        }

        # Criação do Mapa Base
        if estilo_mapa in ["CartoDB Dark", "OpenStreetMap"]:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=tiles_cfg[estilo_mapa], prefer_canvas=True)
        else:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None, prefer_canvas=True)
            folium.TileLayer(
                tiles=tiles_cfg[estilo_mapa], 
                attr=attr_dict[estilo_mapa], 
                name=estilo_mapa,
                overlay=False,
                control=True
            ).add_to(m)

        # Lógica de Camadas (Calor vs Marcadores)
        if tipo_mapa == "Mapa de Calor":
            # Para mapa de calor usamos todos os dados filtrados (visualização melhor)
            # Mas se for MUITO grande (>10k), fazemos sample para não travar
            df_heat = df_filt if len(df_filt) < 10000 else df_filt.sample(10000)
            heat_data = [[row["lat"], row["lon"]] for _, row in df_heat.iterrows()]
            
            HeatMap(
                heat_data, 
                radius=15, 
                blur=20, 
                min_opacity=0.4,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(m)
            
        else: # Marcadores
            # Usa FastMarkerCluster para muitos pontos ou Cluster normal para poucos
            if len(df_map) > 2000:
                FastMarkerCluster(df_map[["lat", "lon"]].values.tolist()).add_to(m)
            else:
                cluster = MarkerCluster().add_to(m)
                for _, row in df_map.iterrows():
                    # Define cor baseada em uma coluna 'class' se existir, senão vermelho padrão
                    is_falso = str(row.get("class", "")).lower() == "falso"
                    color = "#00c3ff" if is_falso else "#ff5252"
                    
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=5,
                        popup=f"Bairro: {row['bairro']}<br>Conf: {row['conf']:.2f}",
                        color=color, 
                        fill=True, 
                        fill_opacity=0.8,
                        fill_color=color
                    ).add_to(cluster)

        st_folium(m, width=None, height=580, returned_objects=[])
    else:
        st.warning("Sem dados para exibir com os filtros atuais.")

with col_pizza:
    st.markdown("**Top 5 Bairros**")
    
    if not df_filt.empty:
        # Conta bairros (excluindo 'Não identificado' para o gráfico ficar mais bonito)
        df_pizza = df_filt[df_filt["bairro"] != "Não identificado"]
        
        if not df_pizza.empty:
            top5 = df_pizza["bairro"].value_counts().nlargest(5)
            
            fig_pie = px.pie(
                values=top5.values, 
                names=top5.index,
                color_discrete_sequence=px.colors.sequential.Agsunset,
                hole=0.4
            )
            fig_pie.update_layout(
                template="plotly_dark", 
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=50, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Pequena tabela de resumo
            st.dataframe(
                top5.reset_index().rename(columns={"index": "Bairro", "bairro": "Qtd"}), 
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Apenas bairros não identificados na seleção.")
    else:
        st.info("Dados insuficientes.")

# ============================================================
# 5) GRÁFICOS INFERIORES
# ============================================================
col_bar, col_tab = st.columns(2)

with col_bar:
    st.markdown("**Confiança Média por Bairro (Top 10)**")
    if not df_filt.empty:
        # Agrupa e calcula média
        media_conf = df_filt.groupby("bairro")["conf"].mean().sort_values(ascending=False).head(10)
        
        fig_bar = px.bar(
            x=media_conf.index, 
            y=media_conf.values, 
            template="plotly_dark", 
            color=media_conf.values,
            color_continuous_scale="Bluered",
            labels={'x': 'Bairro', 'y': 'Confiança Média'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with col_tab:
    st.markdown("**Top 20 Detecções (Tabela)**")
    if not df_filt.empty:
        st.dataframe(
            df_filt.nlargest(20, "conf")[["bairro", "conf", "lat", "lon"]], 
            hide_index=True, 
            height=350,
            use_container_width=True
        )

st.success("✅ Sistema Online - Coordenadas e Cache Otimizados")