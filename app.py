# app.py - VERSIÓN CORREGIDA SIN ERRORES
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

# CSS optimizado
st.markdown("""
    <style>
    .main { background-color: #0f1720; color: #ffffff; }
    .block-container { padding: 1rem; max-width: 100%; }
    .stButton>button { background-color: #1f77b4; color: white; border: none; padding: 10px 24px; border-radius: 5px; width: 100%; }
    .correlation-high { background-color: rgba(0, 255, 0, 0.1); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .correlation-medium { background-color: rgba(255, 255, 0, 0.1); padding: 2px 6px; border-radius: 4px; }
    .correlation-low { background-color: rgba(255, 0, 0, 0.1); padding: 2px 6px; border-radius: 4px; }
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
    df['colocacion'] = (df['cap_2024'] * np.random.uniform(0.1, 0.8, n_clients)).astype(int)
    df['recaudo'] = (df['colocacion'] * np.random.uniform(0.3, 0.9, n_clients)).astype(int)
    df['nomina'] = (df['cap_2024'] * np.random.uniform(0.02, 0.3, n_clients)).astype(int)
    df['margen'] = (df['recaudo'] * np.random.uniform(0.1, 0.4, n_clients)).astype(int)
    
    # Asegurar que no haya valores negativos
    numeric_cols = ['cap_2024', 'cap_2025', 'colocacion', 'recaudo', 'nomina', 'margen']
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    
    return df

# -------------------------
# BÚSQUEDA POR ID - SIMPLIFICADA
# -------------------------
def setup_id_search(df):
    """Sistema de búsqueda por ID simplificado"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Búsqueda Rápida por ID")
    
    id_buscado = st.sidebar.text_input("ID Cliente:", placeholder="Ej: CLI00123", key="search_id")
    
    if st.sidebar.button("Buscar", key="search_btn"):
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
            if st.sidebar.button(f"{row['id_cliente']} - {row['nombre_cliente'][:20]}...", key=f"result_{idx}"):
                st.session_state['selected_client'] = row.to_dict()
                st.session_state['show_client_detail'] = True

# -------------------------
# FILTROS - SIMPLIFICADOS
# -------------------------
def setup_complete_filters(df):
    """Sistema de filtros simplificado"""
    st.sidebar.header("🎛️ Filtros Avanzados")
    
    # Filtro 1: Sucursal
    sucursales = ['Todas'] + sorted(df['sucursal'].dropna().unique().tolist())
    sucursal_seleccionada = st.sidebar.selectbox("📍 Sucursal:", sucursales, key="sucursal_filter")
    
    # Aplicar filtro de sucursal
    if sucursal_seleccionada != 'Todas':
        df_filtrado = df[df['sucursal'] == sucursal_seleccionada].copy()
    else:
        df_filtrado = df.copy()
    
    # Filtro 2: Segmento
    segmentos_disponibles = ['Todos'] + sorted(df_filtrado['segmento'].dropna().unique().tolist())
    segmento_seleccionado = st.sidebar.selectbox("📊 Segmento:", segmentos_disponibles, key="segmento_filter")
    
    if segmento_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['segmento'] == segmento_seleccionado]
    
    # Filtro 3: Rango de CAP
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
# ANÁLISIS COMPLETO - CORREGIDO
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
                valor = df[var_key].mean() * 100
                st.metric(f"📈 {var_name}", f"{valor:.2f}%")
            else:
                valor = df[var_key].sum()
                st.metric(f"💰 {var_name}", f"${valor:,.0f}")
    
    # ESTADÍSTICAS DESCRIPTIVAS COMPLETAS
    st.subheader("📋 Estadísticas Descriptivas Completas")
    
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
    for idx, row in stats_df.iterrows():
        if row['Variable'] != 'Crecimiento %':
            if stats_df.at[idx, 'Total'] != 'N/A':
                stats_df.at[idx, 'Total'] = f"${float(stats_df.at[idx, 'Total']):,.0f}"
                stats_df.at[idx, 'Promedio'] = f"${float(stats_df.at[idx, 'Promedio']):,.0f}"
                stats_df.at[idx, 'Máximo'] = f"${float(stats_df.at[idx, 'Máximo']):,.0f}"
                stats_df.at[idx, 'Mínimo'] = f"${float(stats_df.at[idx, 'Mínimo']):,.0f}"
                stats_df.at[idx, 'Desviación'] = f"${float(stats_df.at[idx, 'Desviación']):,.0f}"
        else:
            stats_df.at[idx, 'Promedio'] = f"{float(stats_df.at[idx, 'Promedio']):.2f}%"
    
    st.dataframe(stats_df, use_container_width=True)
    
    # ANÁLISIS DE CORRELACIONES CRUZADAS
    st.subheader("🔄 Análisis de Correlaciones Cruzadas")
    
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
    
    numeric_vars = [var for var in available_vars.keys() if var in df.select_dtypes(include=[np.number]).columns]
    
    if len(numeric_vars) >= 2:
        corr_matrix = df[numeric_vars].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Matriz de Correlación entre Variables Financieras",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            labels=dict(color="Correlación")
        )
        
        fig_corr.update_xaxes(ticktext=[available_vars.get(var, var) for var in numeric_vars],
                             tickvals=list(range(len(numeric_vars))))
        fig_corr.update_yaxes(ticktext=[available_vars.get(var, var) for var in numeric_vars],
                             tickvals=list(range(len(numeric_vars))))
        
        st.plotly_chart(fig_corr, use_container_width=True, key="full_correlation_matrix")
        
        # Análisis interpretativo de correlaciones
        st.subheader("📈 Interpretación de Correlaciones")
        
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
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
        elif abs(correlation) > 0.3:
            interpretacion = "Moderada"
        else:
            interpretacion = "Débil"
        st.metric("Intensidad", interpretacion)
    
    with col_corr_metrics3:
        direccion = "Positiva" if correlation > 0 else "Negativa"
        st.metric("Dirección", direccion)
    
    # Scatter plot de correlación
    fig_scatter = px.scatter(df, x=var_x, y=var_y,
                           title=f"Relación: {label_x} vs {label_y}",
                           trendline="ols",  # Esta línea usa plotly internamente, no statsmodels
                           hover_data=['nombre_cliente', 'sucursal'] if 'nombre_cliente' in df.columns else None,
                           color='segmento' if 'segmento' in df.columns else None)
    
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.6))
    
    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{var_x}_{var_y}")

# -------------------------
# DASHBOARD PRINCIPAL
# -------------------------
def show_main_dashboard(df):
    """Dashboard principal completo"""
    st.header("📊 Dashboard Ejecutivo")
    
    # KPI PRINCIPALES
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
        cap_total_2024 = df['cap_2024'].sum()
        cap_total_2025 = df['cap_2025'].sum()
        
        if cap_total_2024 > 0:
            crecimiento_total = ((cap_total_2025 - cap_total_2024) / cap_total_2024) * 100
        else:
            crecimiento_total = 0
        
        st.metric("Crecimiento Total", f"{crecimiento_total:.1f}%")
    
    with col5:
        total_colocacion = df['colocacion'].sum() if 'colocacion' in df.columns else 0
        st.metric("Colocación Total", f"${total_colocacion:,.0f}")
    
    # MOSTRAR ANÁLISIS GENERAL COMPLETO
    show_comprehensive_analysis(df)

# -------------------------
# FUNCIONES RESTANTES (simplificadas)
# -------------------------
def analyze_variable_complete(df, variable, nombre_variable):
    """Análisis completo de cada variable financiera"""
    st.header(f"📈 Análisis Detallado: {nombre_variable}")
    
    if variable not in df.columns:
        st.warning(f"⚠️ La variable '{variable}' no está disponible")
        return
    
    df_valido = df[df[variable].notna()].copy()
    
    if len(df_valido) == 0:
        st.warning("No hay datos válidos para analizar")
        return
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = df_valido[variable].sum()
        st.metric("Total", f"${total:,.0f}" if variable != 'crec' else f"{total:.2%}")
    
    with col2:
        promedio = df_valido[variable].mean()
        st.metric("Promedio", f"${promedio:,.0f}" if variable != 'crec' else f"{promedio:.2%}")
    
    with col3:
        maximo = df_valido[variable].max()
        st.metric("Máximo", f"${maximo:,.0f}" if variable != 'crec' else f"{maximo:.2%}")
    
    with col4:
        minimo = df_valido[variable].min()
        st.metric("Mínimo", f"${minimo:,.0f}" if variable != 'crec' else f"{minimo:.2%}")
    
    # Gráficos
    st.subheader("📊 Visualizaciones")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_hist = px.histogram(df_valido, x=variable, 
                               title=f"Distribución de {nombre_variable}",
                               nbins=20,
                               color_discrete_sequence=['#1f77b4'])
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_chart2:
        fig_box = px.box(df_valido, y=variable, 
                        title=f"Distribución - {nombre_variable}",
                        color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig_box, use_container_width=True)

def show_comparative_analysis(df):
    """Análisis comparativo entre variables"""
    st.header("🔄 Análisis Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_cols = ['cap_2024', 'cap_2025', 'colocacion', 'recaudo', 'nomina', 'margen']
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        if len(available_numeric) >= 2:
            corr_matrix = df[available_numeric].corr()
            fig_corr = px.imshow(corr_matrix, 
                               title="Matriz de Correlación",
                               color_continuous_scale='RdBu_r',
                               aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        if 'cap_2024' in df.columns and 'cap_2025' in df.columns:
            fig_scatter = px.scatter(df, x='cap_2024', y='cap_2025',
                                   color='segmento',
                                   title="CAP 2024 vs CAP 2025",
                                   hover_data=['nombre_cliente'],
                                   size_max=15)
            st.plotly_chart(fig_scatter, use_container_width=True)

def show_client_detail(client_data):
    """Vista detallada de cliente individual"""
    st.header(f"👤 Perfil Cliente: {client_data['id_cliente']}")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.subheader("📋 Información General")
        st.write(f"**ID:** {client_data['id_cliente']}")
        st.write(f"**Nombre:** {client_data['nombre_cliente']}")
        st.write(f"**Sucursal:** {client_data['sucursal']}")
    
    with col_info2:
        st.subheader("🏢 Ubicación")
        st.write(f"**Oficina:** {client_data['oficina']}")
        st.write(f"**Segmento:** {client_data['segmento']}")
    
    with col_info3:
        st.subheader("📊 Estado")
        crecimiento = client_data['crec'] * 100
        st.write(f"**Crecimiento:** {crecimiento:.1f}%")
        st.write(f"**Diferencia CAP:** ${client_data['diferencia']:,.0f}")
    
    # Métricas financieras
    st.subheader("💰 Métricas Financieras")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    financial_metrics = [
        ('CAP 2024', 'cap_2024', '💰', metrics_col1),
        ('CAP 2025', 'cap_2025', '📈', metrics_col1),
        ('Colocación', 'colocacion', '💵', metrics_col2),
        ('Recaudo', 'recaudo', '🔄', metrics_col2),
        ('Nómina', 'nomina', '👥', metrics_col3),
        ('Margen', 'margen', '📊', metrics_col3),
        ('Diferencia', 'diferencia', '⚖️', metrics_col4),
        ('Crecimiento', 'crec', '🎯', metrics_col4)
    ]
    
    for nombre, campo, emoji, col in financial_metrics:
        if campo in client_data and pd.notna(client_data[campo]):
            valor = client_data[campo]
            if campo == 'crec':
                valor_formateado = f"{valor:.2%}"
            else:
                valor_formateado = f"${valor:,.0f}"
            col.metric(f"{emoji} {nombre}", valor_formateado)

# -------------------------
# APLICACIÓN PRINCIPAL
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
    
    if df is None or len(df) == 0:
        st.error("No se pudieron cargar los datos. Por favor verifica el archivo.")
        return
    
    # Sidebar - Búsqueda y Filtros
    setup_id_search(df)
    df_filtrado = setup_complete_filters(df)
    
    # Navegación principal
    if st.session_state.get('show_client_detail', False) and st.session_state.get('selected_client') is not None:
        show_client_detail(st.session_state['selected_client'])
        if st.button("↩️ Volver al Dashboard Principal"):
            st.session_state['show_client_detail'] = False
            st.session_state['selected_client'] = None
            st.rerun()
    else:
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
                'cap_2024': 'CAP 2024', 'cap_2025': 'CAP 2025', 'diferencia': 'Diferencia CAP',
                'crec': 'Crecimiento (%)', 'colocacion': 'Colocación', 'recaudo': 'Recaudo',
                'nomina': 'Nómina', 'margen': 'Margen'
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
        
        with tab3:
            st.header("📋 Base de Datos Completa")
            st.write(f"**Total de registros mostrados:** {len(df_filtrado)}")
            st.dataframe(df_filtrado, use_container_width=True, height=600)
        
        with tab4:
            show_comparative_analysis(df_filtrado)

if __name__ == "__main__":
    main()