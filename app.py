# app.py - VERSIÓN MEJORADA CON ANÁLISIS COMPLETO DE VARIABLES Y CORRELACIONES
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración para Render
st.set_page_config(
    page_title="Análisis Financiero - Datos Normalizados",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS optimizado pero completo
st.markdown("""
    <style>
    .main { 
        background-color: #0f1720; 
        color: #ffffff; 
        font-family: 'Arial', sans-serif;
    }
    .block-container { 
        padding: 1rem; 
        max-width: 100%;
    }
    .analysis-box { 
        background: rgba(255,255,255,0.03); 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card { 
        background: rgba(255,255,255,0.05); 
        padding: 15px; 
        border-radius: 8px; 
        margin: 5px; 
        text-align: center;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        width: 100%;
    }
    .stSelectbox>div>div>select {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    .search-box {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .correlation-high {
        background-color: rgba(0, 255, 0, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    .correlation-medium {
        background-color: rgba(255, 255, 0, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .correlation-low {
        background-color: rgba(255, 0, 0, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# CARGAR Y NORMALIZAR DATOS
# -------------------------
@st.cache_data(ttl=3600)
def load_and_normalize_data():
    """Carga y normaliza datos con manejo robusto de errores"""
    try:
        # Intentar cargar archivo real
        df = pd.read_csv("Base_Completa_zona.csv")
        st.success("✅ Archivo de datos cargado correctamente")
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Limitar datos para Render (más eficiente)
        df = df.head(200).copy()
        
        # Filtrar filas válidas
        df = df[
            df['sucursal'].notna() & 
            df['sucursal'].astype(str).str.strip().ne('') &
            ~df['sucursal'].astype(str).str.lower().str.contains('total')
        ].copy()
        
        # Normalización de texto
        df['sucursal'] = df['sucursal'].str.strip().str.title()
        
        text_columns = ['oficina', 'segmento', 'nombre_cliente']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Limpiar ID_cliente
        if 'id_cliente' in df.columns:
            df['id_cliente'] = df['id_cliente'].astype(str).str.strip()
        
        # Limpiar valores numéricos
        numeric_columns = ['cap_2024', 'cap_2025', 'diferencia', 'crec', 'colocacion', 'recaudo', 'nomina', 'margen']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except FileNotFoundError:
        st.warning("📁 No se encontró el archivo CSV. Generando datos de ejemplo...")
        return generate_sample_data()
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.info("📋 Usando datos de ejemplo para continuar...")
        return generate_sample_data()

def generate_sample_data():
    """Genera datos de ejemplo completos y realistas"""
    np.random.seed(42)
    n_clients = 150
    
    data = {
        'id_cliente': [f'CLI{str(i).zfill(5)}' for i in range(1, n_clients+1)],
        'sucursal': np.random.choice(['Cesar', 'Magdalena', 'Guajira', 'Atlantico', 'Bolivar'], n_clients, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'oficina': np.random.choice(['Oficina Central', 'Oficina Norte', 'Oficina Sur', 'Oficina Este'], n_clients),
        'segmento': np.random.choice(['Corporate', 'Personal', 'PYME', 'Empresarial'], n_clients, p=[0.4, 0.3, 0.2, 0.1]),
        'nombre_cliente': [f'Cliente Corporativo {i}' if i % 3 == 0 else f'Cliente Personal {i}' if i % 3 == 1 else f'Empresa PYME {i}' for i in range(1, n_clients+1)],
        'cap_2024': np.random.lognormal(14, 1.2, n_clients).astype(int),
        'cap_2025': np.random.lognormal(14.3, 1.1, n_clients).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Calcular métricas derivadas
    df['diferencia'] = df['cap_2025'] - df['cap_2024']
    df['crec'] = (df['diferencia'] / df['cap_2024']).round(4)
    
    # Métricas adicionales realistas con correlaciones simuladas
    # Colocación correlacionada con CAP
    df['colocacion'] = (df['cap_2024'] * np.random.uniform(0.1, 0.8, n_clients) + 
                        np.random.normal(0, 10000, n_clients)).astype(int)
    
    # Recaudo correlacionado con colocación
    df['recaudo'] = (df['colocacion'] * np.random.uniform(0.3, 0.9, n_clients) + 
                     np.random.normal(0, 5000, n_clients)).astype(int)
    
    # Nómina correlacionada con CAP
    df['nomina'] = (df['cap_2024'] * np.random.uniform(0.02, 0.3, n_clients) + 
                    np.random.normal(0, 2000, n_clients)).astype(int)
    
    # Margen correlacionado con recaudo y nómina
    df['margen'] = (df['recaudo'] * np.random.uniform(0.1, 0.4, n_clients) - 
                    df['nomina'] * np.random.uniform(0.05, 0.2, n_clients) + 
                    np.random.normal(0, 3000, n_clients)).astype(int)
    
    # Asegurar que no haya valores negativos
    numeric_cols = ['cap_2024', 'cap_2025', 'colocacion', 'recaudo', 'nomina', 'margen']
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    
    return df

# -------------------------
# ANÁLISIS GENERAL COMPLETO - MEJORADO
# -------------------------
def show_comprehensive_analysis(df):
    """Análisis general completo con todas las variables y correlaciones"""
    st.header("📊 Análisis General Completo")
    
    # DEFINIR TODAS LAS VARIABLES FINANCIERAS
    financial_variables = {
        'cap_2024': 'CAP 2024', 
        'cap_2025': 'CAP 2025',
        'diferencia': 'Diferencia CAP',
        'crec': 'Crecimiento %',
        'colocacion': 'Colocación',
        'recaudo': 'Recaudo',
        'nomina': 'Nómina',
        'margen': 'Margen'
    }
    
    # Filtrar solo las variables disponibles
    available_vars = {k: v for k, v in financial_variables.items() if k in df.columns}
    
    # KPI PRINCIPALES - TODAS LAS VARIABLES
    st.subheader("🎯 KPIs Principales - Todas las Variables")
    
    # Crear métricas para cada variable
    num_cols = 4
    cols = st.columns(num_cols)
    
    for idx, (var_key, var_name) in enumerate(available_vars.items()):
        col_idx = idx % num_cols
        with cols[col_idx]:
            if var_key == 'crec':
                # Para crecimiento, calcular promedio
                valor = df[var_key].mean() * 100
                st.metric(f"📈 {var_name}", f"{valor:.2f}%")
            else:
                # Para otras variables, calcular suma
                valor = df[var_key].sum()
                st.metric(f"💰 {var_name}", f"${valor:,.0f}")
    
    # ESTADÍSTICAS DESCRIPTIVAS COMPLETAS
    st.subheader("📋 Estadísticas Descriptivas Completas")
    
    # Calcular estadísticas para cada variable
    stats_data = []
    for var_key, var_name in available_vars.items():
        if var_key in df.columns:
            stats = {
                'Variable': var_name,
                'Clientes': df[var_key].count(),
                'Total': df[var_key].sum() if var_key != 'crec' else 'N/A',
                'Promedio': df[var_key].mean() if var_key != 'crec' else df[var_key].mean() * 100,
                'Máximo': df[var_key].max(),
                'Mínimo': df[var_key].min(),
                'Desviación': df[var_key].std()
            }
            stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Formatear números para mejor visualización
    numeric_cols_stats = ['Total', 'Promedio', 'Máximo', 'Mínimo', 'Desviación']
    for col in numeric_cols_stats:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(
                lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) and x != 'N/A' and abs(x) >= 1000 
                else f"{x:.2f}%" if col == 'Promedio' and 'crec' in stats_df.loc[stats_df[col]==x, 'Variable'].values 
                else f"{x:.2f}" if isinstance(x, (int, float)) and x != 'N/A' 
                else x
            )
    
    st.dataframe(stats_df, use_container_width=True)
    
    # ANÁLISIS DE CORRELACIONES CRUZADAS
    st.subheader("🔄 Análisis de Correlaciones Cruzadas")
    
    # Selección de variables para análisis de correlación
    st.write("**Selecciona variables para análisis de correlación:**")
    
    col_corr1, col_corr2 = st.columns(2)
    
    with col_corr1:
        var_x = st.selectbox(
            "Variable X:",
            options=list(available_vars.keys()),
            format_func=lambda x: available_vars[x],
            key="corr_x"
        )
    
    with col_corr2:
        var_y = st.selectbox(
            "Variable Y:",
            options=list(available_vars.keys()),
            format_func=lambda x: available_vars[x],
            index=1 if len(available_vars) > 1 else 0,
            key="corr_y"
        )
    
    if var_x and var_y and var_x != var_y:
        show_correlation_analysis(df, var_x, var_y, available_vars[var_x], available_vars[var_y])
    
    # MATRIZ DE CORRELACIÓN COMPLETA
    st.subheader("🔗 Matriz de Correlación Completa")
    
    # Seleccionar solo variables numéricas para correlación
    numeric_vars = [var for var in available_vars.keys() if var in df.select_dtypes(include=[np.number]).columns]
    
    if len(numeric_vars) >= 2:
        corr_matrix = df[numeric_vars].corr()
        
        # Crear heatmap de correlación
        fig_corr = px.imshow(
            corr_matrix,
            title="Matriz de Correlación entre Variables Financieras",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            labels=dict(color="Correlación")
        )
        
        # Mejorar el formato de los ejes
        fig_corr.update_xaxes(ticktext=[available_vars.get(var, var) for var in numeric_vars],
                             tickvals=list(range(len(numeric_vars))))
        fig_corr.update_yaxes(ticktext=[available_vars.get(var, var) for var in numeric_vars],
                             tickvals=list(range(len(numeric_vars))))
        
        st.plotly_chart(fig_corr, use_container_width=True, key="full_correlation_matrix")
        
        # Análisis interpretativo de correlaciones
        st.subheader("📈 Interpretación de Correlaciones")
        
        # Encontrar correlaciones fuertes
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Correlación fuerte
                    var1 = available_vars.get(numeric_vars[i], numeric_vars[i])
                    var2 = available_vars.get(numeric_vars[j], numeric_vars[j])
                    strong_correlations.append((var1, var2, corr_value))
        
        if strong_correlations:
            st.write("**🔍 Correlaciones Fuertes Identificadas:**")
            for var1, var2, corr in strong_correlations:
                correlation_class = "correlation-high" if abs(corr) > 0.7 else "correlation-medium"
                st.markdown(f"- **{var1}** ↔ **{var2}**: <span class='{correlation_class}'>r = {corr:.3f}</span>", 
                           unsafe_allow_html=True)
        else:
            st.info("No se encontraron correlaciones fuertes (|r| > 0.5) entre las variables.")
    
    # DISTRIBUCIÓN DE CLIENTES POR VARIABLE
    st.subheader("📊 Distribución de Clientes por Variables")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Histograma interactivo
        var_hist = st.selectbox(
            "Variable para histograma:",
            options=list(available_vars.keys()),
            format_func=lambda x: available_vars[x],
            key="hist_var"
        )
        
        if var_hist:
            fig_hist = px.histogram(df, x=var_hist, 
                                   title=f"Distribución de {available_vars[var_hist]}",
                                   nbins=20,
                                   color_discrete_sequence=['#1f77b4'])
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True, key=f"dist_hist_{var_hist}")
    
    with col_dist2:
        # Box plot comparativo por segmento
        var_box = st.selectbox(
            "Variable para análisis por segmento:",
            options=list(available_vars.keys()),
            format_func=lambda x: available_vars[x],
            key="box_var"
        )
        
        if var_box and 'segmento' in df.columns:
            fig_box = px.box(df, x='segmento', y=var_box,
                           title=f"{available_vars[var_box]} por Segmento",
                           color='segmento')
            st.plotly_chart(fig_box, use_container_width=True, key=f"dist_box_{var_box}")

def show_correlation_analysis(df, var_x, var_y, label_x, label_y):
    """Análisis detallado de correlación entre dos variables"""
    
    st.markdown(f"**🔍 Análisis de Correlación: {label_x} vs {label_y}**")
    
    # Calcular correlación
    correlation = df[var_x].corr(df[var_y])
    
    col_corr_metrics1, col_corr_metrics2, col_corr_metrics3 = st.columns(3)
    
    with col_corr_metrics1:
        st.metric("Coeficiente de Correlación", f"{correlation:.3f}")
    
    with col_corr_metrics2:
        # Interpretación de la correlación
        if abs(correlation) > 0.7:
            interpretacion = "Fuerte"
            color = "green"
        elif abs(correlation) > 0.3:
            interpretacion = "Moderada"
            color = "orange"
        else:
            interpretacion = "Débil"
            color = "red"
        st.metric("Intensidad", interpretacion)
    
    with col_corr_metrics3:
        direccion = "Positiva" if correlation > 0 else "Negativa"
        st.metric("Dirección", direccion)
    
    # Scatter plot de correlación
    fig_scatter = px.scatter(df, x=var_x, y=var_y,
                           title=f"Relación: {label_x} vs {label_y}",
                           trendline="ols",
                           hover_data=['nombre_cliente', 'sucursal'] if 'nombre_cliente' in df.columns else None,
                           color='segmento' if 'segmento' in df.columns else None)
    
    # Añadir línea de tendencia
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.6),
                            selector=dict(mode='markers'))
    
    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{var_x}_{var_y}")
    
    # Análisis interpretativo
    st.markdown("**📈 Interpretación:**")
    if correlation > 0.7:
        st.success(f"Existe una fuerte correlación positiva entre {label_x} y {label_y}. "
                  f"Esto sugiere que cuando una variable aumenta, la otra tiende a aumentar también.")
    elif correlation > 0.3:
        st.info(f"Existe una correlación moderada entre {label_x} y {label_y}. "
               f"Hay una relación discernible pero no extremadamente fuerte.")
    elif correlation > -0.3:
        st.warning(f"La correlación entre {label_x} y {label_y} es débil. "
                  f"Las variables parecen tener poca relación lineal directa.")
    else:
        st.error(f"Existe una correlación negativa entre {label_x} y {label_y}. "
                f"Cuando una variable aumenta, la otra tiende a disminuir.")

# -------------------------
# (Mantener las demás funciones igual: setup_id_search, setup_complete_filters, etc.)
# -------------------------

def setup_id_search(df):
    """Sistema de búsqueda por ID mejorado"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Búsqueda Rápida por ID")
    
    col_search1, col_search2 = st.sidebar.columns([2, 1])
    
    with col_search1:
        id_buscado = st.text_input("ID Cliente:", placeholder="Ej: CLI00123", key="search_id")
    
    with col_search2:
        st.write("")  # Espacio
        if st.button("Buscar", key="search_btn"):
            if id_buscado and id_buscado.strip():
                resultados = df[df['id_cliente'].astype(str).str.contains(id_buscado.strip(), case=False, na=False)]
                if len(resultados) > 0:
                    st.session_state['search_results'] = resultados
                    st.session_state['show_search_results'] = True
                else:
                    st.session_state['search_results'] = None
                    st.session_state['show_search_results'] = False
                    st.sidebar.warning("❌ No encontrado")
    
    # Mostrar resultados de búsqueda
    if st.session_state.get('show_search_results', False) and st.session_state.get('search_results') is not None:
        resultados = st.session_state['search_results']
        st.sidebar.success(f"✅ {len(resultados)} cliente(s) encontrado(s)")
        
        for idx, row in resultados.iterrows():
            btn_label = f"{row['id_cliente']} - {row['nombre_cliente'][:20]}..."
            if st.sidebar.button(btn_label, key=f"result_{row['id_cliente']}"):
                st.session_state['selected_client'] = row
                st.session_state['show_client_detail'] = True

def setup_complete_filters(df):
    """Sistema de filtros completo y optimizado"""
    st.sidebar.header("🎛️ Filtros Avanzados")
    
    # Filtro 1: Sucursal
    sucursales = ['Todas'] + sorted(df['sucursal'].dropna().unique().tolist())
    sucursal_seleccionada = st.sidebar.selectbox(
        "📍 Sucursal:",
        sucursales,
        key="sucursal_filter"
    )
    
    # Aplicar filtro de sucursal
    if sucursal_seleccionada != 'Todas':
        df_filtrado = df[df['sucursal'] == sucursal_seleccionada].copy()
    else:
        df_filtrado = df.copy()
    
    # Filtro 2: Oficina (dependiente de sucursal)
    if len(df_filtrado) > 0:
        oficinas_disponibles = ['Todas'] + sorted(df_filtrado['oficina'].dropna().unique().tolist())
    else:
        oficinas_disponibles = ['Todas']
    
    oficina_seleccionada = st.sidebar.selectbox(
        "🏢 Oficina:",
        oficinas_disponibles,
        key="oficina_filter"
    )
    
    if oficina_seleccionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['oficina'] == oficina_seleccionada]
    
    # Filtro 3: Segmento
    if len(df_filtrado) > 0:
        segmentos_disponibles = ['Todos'] + sorted(df_filtrado['segmento'].dropna().unique().tolist())
    else:
        segmentos_disponibles = ['Todos']
    
    segmento_seleccionado = st.sidebar.selectbox(
        "📊 Segmento:",
        segmentos_disponibles,
        key="segmento_filter"
    )
    
    if segmento_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['segmento'] == segmento_seleccionado]
    
    # Filtro 4: Rango de CAP
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Filtros por CAP")
    
    cap_min, cap_max = st.sidebar.slider(
        "Rango CAP 2025:",
        min_value=float(df['cap_2025'].min()),
        max_value=float(df['cap_2025'].max()),
        value=(float(df['cap_2025'].min()), float(df['cap_2025'].max())),
        key="cap_filter"
    )
    
    df_filtrado = df_filtrado[(df_filtrado['cap_2025'] >= cap_min) & (df_filtrado['cap_2025'] <= cap_max)]
    
    return df_filtrado

# -------------------------
# DASHBOARD PRINCIPAL - ACTUALIZADO
# -------------------------
def show_main_dashboard(df):
    """Dashboard principal completo - ACTUALIZADO"""
    st.header("📊 Dashboard Ejecutivo")
    
    # KPI PRINCIPALES - CÁLCULOS CORREGIDOS
    st.subheader("🎯 KPIs Principales")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_clientes = len(df)
        st.metric("Total Clientes", f"{total_clientes:,}")
    
    with col2:
        cap_2024_total = df['cap_2024'].sum()
        st.metric("CAP 2024 Total", f"${cap_2024_total:,.0f}")
    
    with col3:
        cap_2025_total = df['cap_2025'].sum()
        st.metric("CAP 2025 Total", f"${cap_2025_total:,.0f}")
    
    with col4:
        # ✅ CÁLCULO ROBUSTO DEL CRECIMIENTO
        cap_total_2024 = df['cap_2024'].sum()
        cap_total_2025 = df['cap_2025'].sum()
        
        if cap_total_2024 > 0:
            crecimiento_total = ((cap_total_2025 - cap_total_2024) / cap_total_2024) * 100
        else:
            crecimiento_total = 0
        
        # Validar que el crecimiento sea razonable
        if abs(crecimiento_total) > 1000:  # Si es mayor a 1000%, probable error
            # Intentar cálculo alternativo con promedio de crecimientos individuales
            crecimiento_promedio = df['crec'].mean() * 100
            if abs(crecimiento_promedio) < 1000:  # Si este es razonable, usarlo
                crecimiento_total = crecimiento_promedio
            else:
                crecimiento_total = 0  # Fallback
        
        st.metric("Crecimiento Total", f"{crecimiento_total:.1f}%")
    
    with col5:
        total_colocacion = df['colocacion'].sum() if 'colocacion' in df.columns else 0
        st.metric("Colocación Total", f"${total_colocacion:,.0f}")
    
    # MOSTRAR ANÁLISIS GENERAL COMPLETO
    show_comprehensive_analysis(df)

# -------------------------
# (Mantener las demás funciones igual...)
# -------------------------

def analyze_variable_complete(df, variable, nombre_variable):
    """Análisis completo de cada variable financiera"""
    # ... (mantener esta función igual que en tu código original)

def show_comparative_analysis(df):
    """Análisis comparativo entre variables"""
    # ... (mantener esta función igual que en tu código original)

def show_client_detail(client_data):
    """Vista detallada de cliente individual"""
    # ... (mantener esta función igual que en tu código original)

# -------------------------
# APLICACIÓN PRINCIPAL - ACTUALIZADA
# -------------------------
def main():
    # Inicializar session state
    if 'show_client_detail' not in st.session_state:
        st.session_state['show_client_detail'] = False
    if 'selected_client' not in st.session_state:
        st.session_state['selected_client'] = None
    if 'show_search_results' not in st.session_state:
        st.session_state['show_search_results'] = False
    
    # Título principal
    st.title("🏦 Análisis Financiero Completo")
    st.markdown("Sistema avanzado de análisis de datos financieros con capacidades completas de EDA")
    
    # Cargar datos
    with st.spinner("🔄 Cargando y procesando datos..."):
        df = load_and_normalize_data()
    
    if df is None:
        st.error("No se pudieron cargar los datos. Por favor verifica el archivo.")
        return
    
    # Sidebar - Búsqueda y Filtros
    setup_id_search(df)
    df_filtrado = setup_complete_filters(df)
    
    # Navegación principal
    if st.session_state.get('show_client_detail', False) and st.session_state.get('selected_client') is not None:
        # Vista de detalle de cliente
        show_client_detail(st.session_state['selected_client'])
        if st.button("↩️ Volver al Dashboard Principal"):
            st.session_state['show_client_detail'] = False
            st.session_state['selected_client'] = None
            st.rerun()
    else:
        # Dashboard principal con pestañas
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard Principal", 
            "🔍 Análisis por Variable", 
            "📋 Datos Completos",
            "📈 Análisis Comparativo"
        ])
        
        with tab1:
            show_main_dashboard(df_filtrado)
        
        with tab2:
            st.header("📊 Análisis Individual por Variable")
            
            variables_completas = {
                'cap_2024': 'CAP 2024',
                'cap_2025': 'CAP 2025', 
                'diferencia': 'Diferencia CAP',
                'crec': 'Crecimiento (%)',
                'colocacion': 'Colocación',
                'recaudo': 'Recaudo',
                'nomina': 'Nómina',
                'margen': 'Margen'
            }
            
            variables_disponibles = {k: v for k, v in variables_completas.items() if k in df_filtrado.columns}
            
            if variables_disponibles:
                variable_seleccionada = st.selectbox(
                    "Selecciona la variable a analizar:",
                    options=list(variables_disponibles.keys()),
                    format_func=lambda x: variables_disponibles[x],
                    key="var_selector"
                )
                
                if variable_seleccionada:
                    analyze_variable_complete(df_filtrado, variable_seleccionada, variables_disponibles[variable_seleccionada])
            else:
                st.warning("No hay variables disponibles para análisis")
        
        with tab3:
            st.header("📋 Base de Datos Completa")
            st.write(f"**Total de registros mostrados:** {len(df_filtrado)}")
            
            # Selección de columnas a mostrar
            columnas_base = ['id_cliente', 'sucursal', 'oficina', 'segmento', 'nombre_cliente']
            columnas_financieras = [col for col in ['cap_2024', 'cap_2025', 'diferencia', 'crec', 'colocacion', 'recaudo', 'nomina', 'margen'] 
                                  if col in df_filtrado.columns]
            
            columnas_seleccionadas = st.multiselect(
                "Selecciona columnas a mostrar:",
                options=columnas_base + columnas_financieras,
                default=columnas_base + columnas_financieras[:4],
                key="column_selector"
            )
            
            if columnas_seleccionadas:
                df_display = df_filtrado[columnas_seleccionadas].copy()
                
                # Formatear números para mejor visualización
                for col in df_display.select_dtypes(include=[np.number]).columns:
                    if col == 'crec':
                        df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                    else:
                        df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and abs(x) >= 1000 else f"${x:.2f}" if pd.notna(x) else "N/A")
                
                st.dataframe(df_display, use_container_width=True, height=600)
        
        with tab4:
            show_comparative_analysis(df_filtrado)
    
    # Footer informativo

if __name__ == "__main__":
    main()