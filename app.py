# app.py - EDA para Render - Datos Normalizados
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuración para Render
st.set_page_config(
    page_title="Análisis Financiero - Datos Normalizados",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS optimizado
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
    </style>
""", unsafe_allow_html=True)

# -------------------------
# CARGAR Y NORMALIZAR DATOS - OPTIMIZADO PARA RENDER
# -------------------------
@st.cache_data(ttl=3600)  # Cache por 1 hora para Render
def load_and_normalize_data():
    """Carga y normaliza el dataset corrigiendo inconsistencias"""
    try:
        # Cargar datos
        df = pd.read_csv("Base_Completa_zona.csv")
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Tomar solo las filas con datos reales (primeras 100)
        df = df.head(100).copy()
        
        # Filtrar filas válidas
        df = df[
            df['sucursal'].notna() & 
            df['sucursal'].astype(str).str.strip().ne('') &
            ~df['sucursal'].astype(str).str.lower().str.contains('total')
        ].copy()
        
        # 🔥 NORMALIZACIÓN: Corregir "cesar" a "Cesar"
        df['sucursal'] = df['sucursal'].str.strip().str.title()
        
        # Normalizar otras columnas de texto
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
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'Base_Completa_zona.csv'")
        st.info("""
        **Solución para Render:**
        1. Asegúrate de que el archivo esté en el repositorio de GitHub
        2. Verifica que el nombre sea exactamente 'Base_Completa_zona.csv'
        3. En Render, el archivo debe estar en la raíz del proyecto
        """)
        return None
    except Exception as e:
        st.error(f"❌ Error cargando los datos: {e}")
        return None

# -------------------------
# SISTEMA DE FILTROS INTERACTIVOS - OPTIMIZADO
# -------------------------
def setup_filters(df):
    """Configura el sistema de filtros en cascada"""
    
    st.sidebar.header("🎛️ Filtros Interactivos")
    st.sidebar.info("✅ Datos normalizados: 'cesar' → 'Cesar'")
    
    # Filtro 1: Sucursal
    sucursales = ['Todas'] + sorted(df['sucursal'].dropna().unique().tolist())
    sucursal_seleccionada = st.sidebar.selectbox(
        "📍 Selecciona Sucursal:",
        sucursales,
        key="sucursal"
    )
    
    # Filtrar datos basado en Sucursal
    if sucursal_seleccionada != 'Todas':
        df_filtrado = df[df['sucursal'] == sucursal_seleccionada].copy()
    else:
        df_filtrado = df.copy()
    
    # Filtro 2: Oficina
    if len(df_filtrado) > 0:
        oficinas_disponibles = ['Todas'] + sorted(df_filtrado['oficina'].dropna().unique().tolist())
    else:
        oficinas_disponibles = ['Todas']
    
    oficina_seleccionada = st.sidebar.selectbox(
        "🏢 Selecciona Oficina:",
        oficinas_disponibles,
        key="oficina"
    )
    
    if oficina_seleccionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['oficina'] == oficina_seleccionada]
    
    # Filtro 3: Segmento
    if len(df_filtrado) > 0:
        segmentos_disponibles = ['Todos'] + sorted(df_filtrado['segmento'].dropna().unique().tolist())
    else:
        segmentos_disponibles = ['Todos']
    
    segmento_seleccionado = st.sidebar.selectbox(
        "📊 Selecciona Segmento:",
        segmentos_disponibles,
        key="segmento"
    )
    
    if segmento_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['segmento'] == segmento_seleccionado]
    
    # Filtro 4: Cliente específico
    if len(df_filtrado) > 0:
        opciones_clientes = ['Todos']
        for idx, row in df_filtrado.iterrows():
            if pd.notna(row['id_cliente']) and row['id_cliente'] not in ['', 'nan']:
                opcion = f"{row['id_cliente']} - {row['nombre_cliente']}"
            else:
                opcion = row['nombre_cliente']
            opciones_clientes.append(opcion)
        
        cliente_seleccionado = st.sidebar.selectbox(
            "👤 Selecciona Cliente:",
            opciones_clientes,
            key="cliente"
        )
        
        if cliente_seleccionado != 'Todos':
            if ' - ' in cliente_seleccionado:
                id_seleccionado = cliente_seleccionado.split(' - ')[0]
                df_filtrado = df_filtrado[df_filtrado['id_cliente'] == id_seleccionado]
            else:
                df_filtrado = df_filtrado[df_filtrado['nombre_cliente'] == cliente_seleccionado]
    
    return df_filtrado

# -------------------------
# ANÁLISIS DE DATOS - OPTIMIZADO PARA RENDER
# -------------------------
def show_filtered_analysis(df_filtrado, df_original):
    """Muestra análisis basado en los filtros seleccionados"""
    
    st.header("📈 Análisis de Datos Filtrados")
    
    # Información del filtro
    st.write(f"**Mostrando:** {len(df_filtrado)} de {len(df_original)} clientes")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Clientes", len(df_filtrado))
    
    with col2:
        total_cap_2024 = df_filtrado['cap_2024'].sum() if 'cap_2024' in df_filtrado.columns else 0
        st.metric("💰 CAP 2024", f"${total_cap_2024:,.0f}")
    
    with col3:
        total_cap_2025 = df_filtrado['cap_2025'].sum() if 'cap_2025' in df_filtrado.columns else 0
        st.metric("💰 CAP 2025", f"${total_cap_2025:,.0f}")
    
    with col4:
        crecimiento_total = (total_cap_2025 - total_cap_2024) / total_cap_2024 if total_cap_2024 > 0 else 0
        st.metric("📈 Crecimiento", f"{crecimiento_total:.2%}")
    
    # Datos filtrados
    st.subheader("📋 Datos Filtrados")
    
    columnas_a_mostrar = ['id_cliente', 'sucursal', 'oficina', 'segmento', 'nombre_cliente', 
                         'cap_2024', 'cap_2025', 'diferencia', 'crec']
    
    columnas_adicionales = ['colocacion', 'recaudo', 'nomina', 'margen']
    for col in columnas_adicionales:
        if col in df_filtrado.columns:
            columnas_a_mostrar.append(col)
    
    df_mostrar = df_filtrado[columnas_a_mostrar].copy()
    
    # Formatear números
    for col in df_mostrar.select_dtypes(include=[np.number]).columns:
        if col not in ['id_cliente', 'crec']:
            df_mostrar[col] = df_mostrar[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and abs(x) >= 1000 else f"{x:.2f}" if pd.notna(x) else "N/A")
        elif col == 'crec':
            df_mostrar[col] = df_mostrar[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    
    st.dataframe(df_mostrar, use_container_width=True, height=400)
    
    # Análisis gráfico
    if len(df_filtrado) > 1:
        show_multiple_analysis(df_filtrado)
    elif len(df_filtrado) == 1:
        show_single_client_analysis(df_filtrado.iloc[0])

def show_multiple_analysis(df_filtrado):
    """Análisis para múltiples clientes"""
    
    st.subheader("📊 Análisis Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de CAP por sucursal
        cap_por_sucursal = df_filtrado.groupby('sucursal')[['cap_2024', 'cap_2025']].sum().reset_index()
        fig = px.bar(cap_por_sucursal, x='sucursal', y=['cap_2024', 'cap_2025'],
                    title="CAP por Sucursal", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de segmentos
        segmento_counts = df_filtrado['segmento'].value_counts()
        fig = px.pie(values=segmento_counts.values, names=segmento_counts.index,
                    title="Distribución por Segmento")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top clientes
    if len(df_filtrado) > 5:
        st.subheader("🏆 Top 10 Clientes por CAP 2025")
        top_clientes = df_filtrado.nlargest(10, 'cap_2025')
        top_clientes['etiqueta'] = top_clientes['id_cliente'] + ' - ' + top_clientes['nombre_cliente'].str.slice(0, 30)
        
        fig = px.bar(top_clientes, x='etiqueta', y='cap_2025',
                    title="Top 10 Clientes", labels={'etiqueta': 'Cliente'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def show_single_client_analysis(cliente):
    """Análisis para cliente individual"""
    
    st.subheader(f"👤 Análisis Detallado: {cliente['id_cliente']} - {cliente['nombre_cliente']}")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write(f"**ID Cliente:** {cliente['id_cliente']}")
        st.write(f"**Sucursal:** {cliente['sucursal']}")
    
    with col_info2:
        st.write(f"**Oficina:** {cliente['oficina']}")
        st.write(f"**Segmento:** {cliente['segmento']}")
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CAP 2024", f"${cliente['cap_2024']:,.0f}" if pd.notna(cliente['cap_2024']) else "N/A")
        st.metric("CAP 2025", f"${cliente['cap_2025']:,.0f}" if pd.notna(cliente['cap_2025']) else "N/A")
    
    with col2:
        st.metric("Diferencia", f"${cliente['diferencia']:,.0f}" if pd.notna(cliente['diferencia']) else "N/A")
        if pd.notna(cliente['colocacion']):
            st.metric("Colocación", f"${cliente['colocacion']:,.0f}")
    
    with col3:
        if pd.notna(cliente['recaudo']):
            st.metric("Recaudo", f"${cliente['recaudo']:,.0f}")
        if pd.notna(cliente['margen']):
            st.metric("Margen", f"${cliente['margen']:,.0f}")

# -------------------------
# ESTADÍSTICAS GENERALES
# -------------------------
def show_general_stats(df):
    """Estadísticas generales"""
    
    st.header("📊 Estadísticas Generales")
    st.success("✅ Datos normalizados: 'cesar' corregido a 'Cesar'")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏢 Sucursales")
        for sucursal, count in df['sucursal'].value_counts().items():
            st.write(f"**{sucursal}:** {count} clientes")
    
    with col2:
        st.subheader("📈 Segmentos")
        for segmento, count in df['segmento'].value_counts().items():
            st.write(f"**{segmento}:** {count} clientes")
    
    with col3:
        st.subheader("💰 Totales")
        st.metric("Clientes", len(df))
        st.metric("CAP 2024", f"${df['cap_2024'].sum():,.0f}")
        st.metric("CAP 2025", f"${df['cap_2025'].sum():,.0f}")

# -------------------------
# APLICACIÓN PRINCIPAL
# -------------------------
def main():
    st.title("🏦 Análisis Financiero - Datos Normalizados")
    st.markdown("Sistema de análisis interactivo para datos financieros")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_and_normalize_data()
    
    if df is None:
        return
    
    # Sidebar info
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.write(f"**Clientes:** {len(df)}")
    st.sidebar.write(f"**Sucursales:** {len(df['sucursal'].unique())}")
    
    # Filtros
    df_filtrado = setup_filters(df)
    
    # Análisis
    show_filtered_analysis(df_filtrado, df)
    show_general_stats(df)
    
    # Footer para Render
    st.markdown("---")
    st.markdown("**🚀 Desplegado en Render** | *Análisis Financiero v1.0*")

if __name__ == "__main__":
    main()