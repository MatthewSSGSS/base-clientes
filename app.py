# app.py - VERSIÓN CORREGIDA CON KEYS ÚNICOS
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
    
    # Métricas adicionales realistas
    df['colocacion'] = (df['cap_2024'] * np.random.uniform(0.1, 0.8, n_clients)).astype(int)
    df['recaudo'] = (df['cap_2024'] * np.random.uniform(0.05, 0.6, n_clients)).astype(int)
    df['nomina'] = (df['cap_2024'] * np.random.uniform(0.02, 0.3, n_clients)).astype(int)
    df['margen'] = (df['cap_2024'] * np.random.uniform(0.01, 0.2, n_clients)).astype(int)
    
    # Asegurar que no haya valores negativos
    numeric_cols = ['cap_2024', 'cap_2025', 'colocacion', 'recaudo', 'nomina', 'margen']
    for col in numeric_cols:
        df[col] = df[col].abs()
    
    return df

# -------------------------
# BÚSQUEDA POR ID - MEJORADA
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

# -------------------------
# SISTEMA DE FILTROS COMPLETO
# -------------------------
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
# ANÁLISIS POR VARIABLE COMPLETO
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
    
    # MÉTRICAS PRINCIPALES
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
    
    # GRÁFICOS AVANZADOS
    st.subheader("📊 Visualizaciones")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Histograma con percentiles
        fig_hist = px.histogram(df_valido, x=variable, 
                               title=f"Distribución de {nombre_variable}",
                               nbins=20,
                               color_discrete_sequence=['#1f77b4'])
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{variable}")
    
    with col_chart2:
        # Box plot
        fig_box = px.box(df_valido, y=variable, 
                        title=f"Distribución - {nombre_variable}",
                        color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig_box, use_container_width=True, key=f"box_{variable}")
    
    # ANÁLISIS POR SUCURSAL
    st.subheader("🏢 Análisis por Sucursal")
    sucursal_analysis = df_valido.groupby('sucursal').agg({
        variable: ['sum', 'mean', 'count', 'std']
    }).round(2)
    
    sucursal_analysis.columns = ['Total', 'Promedio', 'Clientes', 'Desviación']
    st.dataframe(sucursal_analysis, use_container_width=True)
    
    # TOP 10 CLIENTES
    st.subheader(f"🏆 Top 10 Clientes por {nombre_variable}")
    top_clientes = df_valido.nlargest(10, variable)[['id_cliente', 'nombre_cliente', 'sucursal', variable]].copy()
    
    # Formatear valores para display
    if variable != 'crec':
        top_clientes[variable] = top_clientes[variable].apply(lambda x: f"${x:,.0f}")
    else:
        top_clientes[variable] = top_clientes[variable].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(top_clientes, use_container_width=True)

# -------------------------
# ANÁLISIS COMPARATIVO COMPLETO
# -------------------------
def show_comparative_analysis(df):
    """Análisis comparativo entre variables"""
    st.header("🔄 Análisis Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlación entre variables principales
        numeric_cols = ['cap_2024', 'cap_2025', 'colocacion', 'recaudo', 'nomina', 'margen']
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        if len(available_numeric) >= 2:
            corr_matrix = df[available_numeric].corr()
            fig_corr = px.imshow(corr_matrix, 
                               title="Matriz de Correlación",
                               color_continuous_scale='RdBu_r',
                               aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True, key="correlation_matrix")
    
    with col2:
        # Scatter plot CAP 2024 vs CAP 2025
        if 'cap_2024' in df.columns and 'cap_2025' in df.columns:
            fig_scatter = px.scatter(df, x='cap_2024', y='cap_2025',
                                   color='segmento',
                                   title="CAP 2024 vs CAP 2025",
                                   hover_data=['nombre_cliente'],
                                   size_max=15)
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_cap")

# -------------------------
# DASHBOARD PRINCIPAL
# -------------------------
def show_main_dashboard(df):
    """Dashboard principal completo - CORREGIDO"""
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
    
    # GRÁFICOS PRINCIPALES
    st.subheader("📈 Tendencias Principales")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Evolución por sucursal
        sucursal_evolution = df.groupby('sucursal')[['cap_2024', 'cap_2025']].sum().reset_index()
        fig_evolution = px.bar(sucursal_evolution, x='sucursal', y=['cap_2024', 'cap_2025'],
                              title="Evolución CAP por Sucursal", barmode='group')
        st.plotly_chart(fig_evolution, use_container_width=True, key="evolution_bar")
    
    with col_chart2:
        # Distribución por segmento
        segmento_dist = df['segmento'].value_counts()
        fig_segment = px.pie(values=segmento_dist.values, names=segmento_dist.index,
                           title="Distribución por Segmento de Clientes")
        st.plotly_chart(fig_segment, use_container_width=True, key="segment_pie")
    
    # ANÁLISIS ADICIONAL DE CRECIMIENTO
    st.subheader("📊 Análisis de Crecimiento Detallado")
    
    col_crec1, col_crec2 = st.columns(2)
    
    with col_crec1:
        # Crecimiento por sucursal
        crecimiento_sucursal = df.groupby('sucursal').apply(
            lambda x: ((x['cap_2025'].sum() - x['cap_2024'].sum()) / x['cap_2024'].sum() * 100) if x['cap_2024'].sum() > 0 else 0
        ).round(1)
        
        fig_crec_sucursal = px.bar(x=crecimiento_sucursal.index, y=crecimiento_sucursal.values,
                                  title="Crecimiento por Sucursal (%)",
                                  labels={'x': 'Sucursal', 'y': 'Crecimiento %'})
        st.plotly_chart(fig_crec_sucursal, use_container_width=True, key="growth_by_branch")
    
    with col_crec2:
        # Distribución de crecimiento individual
        crecimiento_individual = df['crec'] * 100
        fig_crec_dist = px.histogram(x=crecimiento_individual, 
                                    title="Distribución de Crecimientos Individuales",
                                    labels={'x': 'Crecimiento %', 'y': 'Número de Clientes'})
        st.plotly_chart(fig_crec_dist, use_container_width=True, key="growth_distribution")

# -------------------------
# VISTA DETALLE CLIENTE
# -------------------------
def show_client_detail(client_data):
    """Vista detallada de cliente individual"""
    st.header(f"👤 Perfil Cliente: {client_data['id_cliente']}")
    
    # Información básica
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
    
    # MÉTRICAS FINANCIERAS
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