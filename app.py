# app.py - ANÁLISIS COMPLETO DE TODAS LAS VARIABLES DEL DATASET REAL
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
    page_title="Análisis Financiero Completo - Dataset Real",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS optimizado
st.markdown("""
    <style>
    .main { background-color: #0f1720; color: #ffffff; }
    .block-container { padding: 1rem; max-width: 100%; }
    .stButton>button { background-color: #1f77b4; color: white; border: none; padding: 10px 24px; border-radius: 5px; width: 100%; }
    .metric-card { 
        background: rgba(255,255,255,0.05); 
        padding: 15px; 
        border-radius: 8px; 
        margin: 5px; 
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .analysis-section { 
        background: rgba(255,255,255,0.03); 
        padding: 20px; 
        border-radius: 10px; 
        margin: 15px 0; 
        border: 1px solid rgba(255,255,255,0.1);
    }
    .data-table { 
        background: rgba(255,255,255,0.02); 
        border-radius: 8px; 
        padding: 10px;
        margin: 10px 0;
    }
    .correlation-high { background-color: rgba(0, 255, 0, 0.1); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .correlation-medium { background-color: rgba(255, 255, 0, 0.1); padding: 2px 6px; border-radius: 4px; }
    .correlation-low { background-color: rgba(255, 0, 0, 0.1); padding: 2px 6px; border-radius: 4px; }
    .variable-category { color: #1f77b4; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# CARGAR Y NORMALIZAR DATOS DEL DATASET REAL
# -------------------------
@st.cache_data(ttl=3600)
def load_and_normalize_data():
    """Carga y normaliza datos del dataset real"""
    try:
        # Cargar archivo real
        df = pd.read_csv("Base_Completa_zona.csv")
        st.success("✅ Archivo de datos cargado correctamente")
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Filtrar filas válidas (eliminar totales y filas vacías)
        df = df[
            (df['sucursal'].notna()) & 
            (df['sucursal'].astype(str).str.strip().ne('')) &
            (~df['sucursal'].astype(str).str.lower().str.contains('total', na=False)) &
            (df['id_cliente'].notna()) &
            (df['id_cliente'].astype(str).str.strip().ne(''))
        ].copy()
        
        # Normalización de texto
        text_columns = ['sucursal', 'oficina', 'segmento', 'nombre_cliente']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Limpiar valores numéricos - TODAS LAS VARIABLES DEL DATASET
        numeric_columns = [
            'cap_2024', 'cap_2025', 'diferencia', 'crec', 
            'colocacion', 'recaudo', 'nomina', 'margen'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Reemplazar valores extremadamente altos/bajos
                if col in ['cap_2024', 'cap_2025', 'colocacion', 'recaudo', 'nomina', 'margen']:
                    # Limitar a valores razonables para análisis
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                df[col] = df[col].fillna(0)
        
        return df
        
    except FileNotFoundError:
        st.warning("📁 No se encontró el archivo CSV. Generando datos de ejemplo...")
        return generate_sample_data()
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.info("📋 Usando datos de ejemplo para continuar...")
        return generate_sample_data()

def generate_sample_data():
    """Genera datos de ejemplo basados en la estructura del dataset real"""
    np.random.seed(42)
    n_clients = 100
    
    data = {
        'id_cliente': [f'CLI{str(i).zfill(5)}' for i in range(1, n_clients+1)],
        'sucursal': np.random.choice(['Cesar', 'Magdalena', 'Guajira', 'Atlantico', 'Bolivar'], n_clients, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'oficina': np.random.choice(['Palmas', 'Principal', 'Buenavista', 'Centro Historico', 'Fundacion'], n_clients),
        'segmento': np.random.choice(['Territorial', 'Descentralizado', 'Hospital', 'Universidad', 'Transito'], n_clients, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'nombre_cliente': [f'Cliente {i}' for i in range(1, n_clients+1)],
        'cap_2024': np.random.lognormal(16, 1.5, n_clients).astype(int),
        'cap_2025': np.random.lognormal(16.2, 1.4, n_clients).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Calcular métricas derivadas
    df['diferencia'] = df['cap_2025'] - df['cap_2024']
    df['crec'] = (df['diferencia'] / df['cap_2024']).round(4)
    
    # Métricas adicionales realistas
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
# ANÁLISIS COMPLETO DE TODAS LAS VARIABLES
# -------------------------
def show_comprehensive_analysis(df):
    """Análisis general completo estructurado en tablas organizadas"""
    st.header("📊 Análisis General Completo - Todas las Variables")
    
    # DEFINIR TODAS LAS VARIABLES DEL DATASET REAL
    financial_variables = {
        'cap_2024': {'nombre': 'CAP 2024', 'categoria': 'Captación', 'tipo': 'monetario'},
        'cap_2025': {'nombre': 'CAP 2025', 'categoria': 'Captación', 'tipo': 'monetario'},
        'diferencia': {'nombre': 'Diferencia CAP', 'categoria': 'Crecimiento', 'tipo': 'monetario'},
        'crec': {'nombre': 'Crecimiento %', 'categoria': 'Crecimiento', 'tipo': 'porcentaje'},
        'colocacion': {'nombre': 'Colocación', 'categoria': 'Operaciones', 'tipo': 'monetario'},
        'recaudo': {'nombre': 'Recaudo', 'categoria': 'Operaciones', 'tipo': 'monetario'},
        'nomina': {'nombre': 'Nómina', 'categoria': 'Gastos', 'tipo': 'monetario'},
        'margen': {'nombre': 'Margen', 'categoria': 'Rentabilidad', 'tipo': 'monetario'}
    }
    
    # Filtrar solo las variables disponibles
    available_vars = {k: v for k, v in financial_variables.items() if k in df.columns}
    
    # SECCIÓN 1: RESUMEN EJECUTIVO COMPLETO
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("🎯 Resumen Ejecutivo - Todas las Variables")
        
        # Organizar variables por categoría
        categorias = {}
        for var_key, var_info in available_vars.items():
            categoria = var_info['categoria']
            if categoria not in categorias:
                categorias[categoria] = []
            categorias[categoria].append((var_key, var_info))
        
        # Mostrar KPIs por categoría
        for categoria, variables in categorias.items():
            st.markdown(f"**📁 {categoria}**")
            cols = st.columns(len(variables))
            
            for idx, (var_key, var_info) in enumerate(variables):
                with cols[idx]:
                    if var_info['tipo'] == 'porcentaje':
                        valor = df[var_key].mean() * 100
                        st.metric(f"📈 {var_info['nombre']}", f"{valor:.2f}%")
                    else:
                        valor = df[var_key].sum()
                        st.metric(f"💰 {var_info['nombre']}", f"${valor:,.0f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 2: ANÁLISIS DETALLADO POR VARIABLE
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📈 Análisis Detallado por Variable")
        
        # Crear tabla de estadísticas completas
        stats_data = []
        for var_key, var_info in available_vars.items():
            if var_key in df.columns:
                if var_info['tipo'] == 'porcentaje':
                    stats = {
                        'Variable': var_info['nombre'],
                        'Categoría': var_info['categoria'],
                        'Tipo': var_info['tipo'],
                        'N° Clientes': df[var_key].count(),
                        'Total': 'N/A',
                        'Promedio': f"{df[var_key].mean() * 100:.2f}%",
                        'Máximo': f"{df[var_key].max() * 100:.2f}%",
                        'Mínimo': f"{df[var_key].min() * 100:.2f}%",
                        'Desviación': f"{df[var_key].std() * 100:.2f}%",
                        'Mediana': f"{df[var_key].median() * 100:.2f}%"
                    }
                else:
                    stats = {
                        'Variable': var_info['nombre'],
                        'Categoría': var_info['categoria'],
                        'Tipo': var_info['tipo'],
                        'N° Clientes': df[var_key].count(),
                        'Total': f"${df[var_key].sum():,.0f}",
                        'Promedio': f"${df[var_key].mean():,.0f}",
                        'Máximo': f"${df[var_key].max():,.0f}",
                        'Mínimo': f"${df[var_key].min():,.0f}",
                        'Desviación': f"${df[var_key].std():,.0f}",
                        'Mediana': f"${df[var_key].median():,.0f}"
                    }
                stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, height=500)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 3: DISTRIBUCIÓN GEOGRÁFICA Y POR SEGMENTO
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("🏢 Distribución Geográfica y por Segmento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Análisis por Sucursal
            st.markdown("**📍 Por Sucursal**")
            sucursal_data = []
            for sucursal in df['sucursal'].unique():
                df_sucursal = df[df['sucursal'] == sucursal]
                sucursal_info = {
                    'Sucursal': sucursal,
                    'N° Clientes': len(df_sucursal),
                    'CAP Total 2025': f"${df_sucursal['cap_2025'].sum():,.0f}",
                    'Colocación Total': f"${df_sucursal['colocacion'].sum():,.0f}" if 'colocacion' in df_sucursal.columns else "N/A",
                    'Crecimiento Prom': f"{df_sucursal['crec'].mean() * 100:.1f}%",
                    'Margen Total': f"${df_sucursal['margen'].sum():,.0f}" if 'margen' in df_sucursal.columns else "N/A"
                }
                sucursal_data.append(sucursal_info)
            
            sucursal_df = pd.DataFrame(sucursal_data)
            st.dataframe(sucursal_df, use_container_width=True, height=300)
        
        with col2:
            # Análisis por Segmento
            st.markdown("**📊 Por Segmento**")
            if 'segmento' in df.columns:
                segmento_data = []
                for segmento in df['segmento'].unique():
                    df_segmento = df[df['segmento'] == segmento]
                    segmento_info = {
                        'Segmento': segmento,
                        'N° Clientes': len(df_segmento),
                        '% del Total': f"{(len(df_segmento) / len(df)) * 100:.1f}%",
                        'CAP Promedio': f"${df_segmento['cap_2025'].mean():,.0f}",
                        'Colocación Prom': f"${df_segmento['colocacion'].mean():,.0f}" if 'colocacion' in df_segmento.columns else "N/A",
                        'Crecimiento Prom': f"{df_segmento['crec'].mean() * 100:.1f}%"
                    }
                    segmento_data.append(segmento_info)
                
                segmento_df = pd.DataFrame(segmento_data)
                st.dataframe(segmento_df, use_container_width=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 4: ANÁLISIS DE CRECIMIENTO Y RENTABILIDAD
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📈 Análisis de Crecimiento y Rentabilidad")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Clientes con mayor crecimiento
            st.markdown("**🚀 Top 10 - Mayor Crecimiento**")
            top_crecimiento = df.nlargest(10, 'crec')[['id_cliente', 'nombre_cliente', 'sucursal', 'segmento', 'crec', 'cap_2024', 'cap_2025']].copy()
            top_crecimiento['crec'] = top_crecimiento['crec'].apply(lambda x: f"{x:.2%}")
            top_crecimiento['cap_2024'] = top_crecimiento['cap_2024'].apply(lambda x: f"${x:,.0f}")
            top_crecimiento['cap_2025'] = top_crecimiento['cap_2025'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_crecimiento, use_container_width=True)
        
        with col2:
            # Clientes con mayor margen
            if 'margen' in df.columns:
                st.markdown("**💰 Top 10 - Mayor Margen**")
                top_margen = df.nlargest(10, 'margen')[['id_cliente', 'nombre_cliente', 'sucursal', 'segmento', 'margen', 'recaudo']].copy()
                top_margen['margen'] = top_margen['margen'].apply(lambda x: f"${x:,.0f}")
                top_margen['recaudo'] = top_margen['recaudo'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_margen, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 5: ANÁLISIS DE OPERACIONES
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("💼 Análisis de Operaciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top colocación
            if 'colocacion' in df.columns:
                st.markdown("**🏆 Top 10 - Mayor Colocación**")
                top_colocacion = df.nlargest(10, 'colocacion')[['id_cliente', 'nombre_cliente', 'sucursal', 'segmento', 'colocacion', 'cap_2025']].copy()
                top_colocacion['colocacion'] = top_colocacion['colocacion'].apply(lambda x: f"${x:,.0f}")
                top_colocacion['cap_2025'] = top_colocacion['cap_2025'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_colocacion, use_container_width=True)
        
        with col2:
            # Top recaudo
            if 'recaudo' in df.columns:
                st.markdown("**🔄 Top 10 - Mayor Recaudo**")
                top_recaudo = df.nlargest(10, 'recaudo')[['id_cliente', 'nombre_cliente', 'sucursal', 'segmento', 'recaudo', 'colocacion']].copy()
                top_recaudo['recaudo'] = top_recaudo['recaudo'].apply(lambda x: f"${x:,.0f}")
                top_recaudo['colocacion'] = top_recaudo['colocacion'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_recaudo, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 6: ANÁLISIS DE CORRELACIONES COMPLETO
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("🔗 Análisis de Correlaciones Completo")
        
        # Matriz de correlación
        numeric_vars = [var for var in available_vars.keys() if var in df.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_vars) >= 2:
            corr_matrix = df[numeric_vars].corr()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Matriz de Correlación entre Todas las Variables",
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    labels=dict(color="Correlación")
                )
                # Mejorar etiquetas
                labels = [available_vars.get(var, {}).get('nombre', var) for var in numeric_vars]
                fig_corr.update_xaxes(ticktext=labels, tickvals=list(range(len(numeric_vars))))
                fig_corr.update_yaxes(ticktext=labels, tickvals=list(range(len(numeric_vars))))
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Correlaciones destacadas
                st.subheader("📊 Correlaciones Destacadas")
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.3:  # Mostrar correlaciones moderadas y fuertes
                            var1 = available_vars.get(numeric_vars[i], {}).get('nombre', numeric_vars[i])
                            var2 = available_vars.get(numeric_vars[j], {}).get('nombre', numeric_vars[j])
                            intensidad = 'Alta' if abs(corr_value) > 0.7 else 'Media' if abs(corr_value) > 0.5 else 'Baja'
                            strong_correlations.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'Correlación': f"{corr_value:.3f}",
                                'Intensidad': intensidad
                            })
                
                if strong_correlations:
                    corr_df = pd.DataFrame(strong_correlations)
                    st.dataframe(corr_df, use_container_width=True, height=400)
                else:
                    st.info("No se encontraron correlaciones significativas (|r| > 0.3)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 7: ANÁLISIS COMPARATIVO ENTRE VARIABLES CLAVE
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📊 Análisis Comparativo entre Variables Clave")
        
        # Selección de variables para comparación
        col1, col2 = st.columns(2)
        
        with col1:
            var_x = st.selectbox(
                "Variable X:",
                options=list(available_vars.keys()),
                format_func=lambda x: available_vars[x]['nombre'],
                key="comp_x"
            )
        
        with col2:
            # Filtrar para que no sea la misma variable
            other_vars = [v for v in available_vars.keys() if v != var_x]
            var_y = st.selectbox(
                "Variable Y:",
                options=other_vars,
                format_func=lambda x: available_vars[x]['nombre'],
                key="comp_y"
            )
        
        if var_x and var_y:
            # Scatter plot comparativo
            fig_scatter = px.scatter(df, x=var_x, y=var_y,
                                   title=f"Relación: {available_vars[var_x]['nombre']} vs {available_vars[var_y]['nombre']}",
                                   hover_data=['nombre_cliente', 'sucursal', 'segmento'],
                                   color='segmento' if 'segmento' in df.columns else None,
                                   size_max=15)
            
            # Calcular y añadir línea de tendencia manualmente
            try:
                x_data = df[var_x].values
                y_data = df[var_y].values
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_valid = x_data[valid_mask]
                y_valid = y_data[valid_mask]
                
                if len(x_valid) > 1:
                    A = np.vstack([x_valid, np.ones(len(x_valid))]).T
                    m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                    
                    x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
                    y_trend = m * x_trend + c
                    
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=x_trend, y=y_trend,
                            mode='lines',
                            name='Línea de Tendencia',
                            line=dict(color='red', width=2, dash='dash')
                        )
                    )
            except Exception as e:
                st.warning(f"No se pudo calcular la línea de tendencia: {e}")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Mostrar estadísticas de correlación
            correlation = df[var_x].corr(df[var_y])
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Coeficiente de Correlación", f"{correlation:.3f}")
            with col_stat2:
                intensidad = "Fuerte" if abs(correlation) > 0.7 else "Moderada" if abs(correlation) > 0.3 else "Débil"
                st.metric("Intensidad", intensidad)
            with col_stat3:
                direccion = "Positiva" if correlation > 0 else "Negativa"
                st.metric("Dirección", direccion)
        
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FUNCIONES RESTANTES (mantenidas igual pero optimizadas)
# -------------------------

def setup_id_search(df):
    """Sistema de búsqueda por ID simplificado"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Búsqueda Rápida por ID")
    
    id_buscado = st.sidebar.text_input("ID Cliente:", placeholder="Ej: 8001039206", key="search_id")
    
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
    
    # Filtro 3: Rango de CAP 2025
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Filtros por CAP 2025")
    
    cap_min, cap_max = st.sidebar.slider(
        "Rango CAP 2025:",
        min_value=float(df['cap_2025'].min()),
        max_value=float(df['cap_2025'].max()),
        value=(float(df['cap_2025'].min()), float(df['cap_2025'].max())),
        key="cap_filter"
    )
    
    df_filtrado = df_filtrado[(df_filtrado['cap_2025'] >= cap_min) & (df_filtrado['cap_2025'] <= cap_max)]
    
    return df_filtrado

def show_main_dashboard(df):
    """Dashboard principal completo"""
    st.header("📊 Dashboard Ejecutivo - Dataset Real")
    
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

# ... (mantener las demás funciones analyze_variable_complete, show_comparative_analysis, show_client_detail, main igual que antes)

# Las funciones analyze_variable_complete, show_comparative_analysis, show_client_detail, y main se mantienen igual que en el código anterior
# Solo asegúrate de que estén presentes en tu archivo

if __name__ == "__main__":
    main()