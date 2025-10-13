# app.py - EDA con normalización de datos y ID de Cliente
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Análisis Financiero - Datos Normalizados", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0f1720; color: #ffffff; }
    .block-container { padding: 1rem; }
    .analysis-box { background: rgba(255,255,255,0.03); padding:15px; border-radius:10px; margin:10px 0; }
    .metric-card { background: rgba(255,255,255,0.05); padding:15px; border-radius:8px; margin:5px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# CARGAR Y NORMALIZAR DATOS
# -------------------------
@st.cache_data
def load_and_normalize_data():
    """Carga y normaliza el dataset corrigiendo inconsistencias"""
    try:
        # Cargar datos
        df = pd.read_csv("Base_Completa_zona.csv")
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # ESTRATEGIA: Tomar las primeras 100 filas que tienen datos
        df = df.head(100).copy()
        
        # Filtrar filas que tienen datos en las columnas clave
        df = df[
            df['sucursal'].notna() & 
            df['sucursal'].astype(str).str.strip().ne('') &
            ~df['sucursal'].astype(str).str.lower().str.contains('total')
        ].copy()
        
        # 🔥 NORMALIZACIÓN CRÍTICA: Corregir "cesar" a "Cesar"
        df['sucursal'] = df['sucursal'].str.strip().str.title()
        
        # También normalizar otras columnas de texto
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
        return None
    except Exception as e:
        st.error(f"❌ Error cargando los datos: {e}")
        return None

# -------------------------
# SISTEMA DE FILTROS INTERACTIVOS
# -------------------------
def setup_filters(df):
    """Configura el sistema de filtros en cascada"""
    
    st.sidebar.header("🎛️ Filtros Interactivos")
    
    # Mostrar información de normalización
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
    
    # Filtro 2: Oficina (depende de Sucursal)
    if len(df_filtrado) > 0:
        oficinas_disponibles = ['Todas'] + sorted(df_filtrado['oficina'].dropna().unique().tolist())
    else:
        oficinas_disponibles = ['Todas']
    
    oficina_seleccionada = st.sidebar.selectbox(
        "🏢 Selecciona Oficina:",
        oficinas_disponibles,
        key="oficina"
    )
    
    # Filtrar basado en Oficina
    if oficina_seleccionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['oficina'] == oficina_seleccionada]
    
    # Filtro 3: Segmento (depende de Oficina)
    if len(df_filtrado) > 0:
        segmentos_disponibles = ['Todos'] + sorted(df_filtrado['segmento'].dropna().unique().tolist())
    else:
        segmentos_disponibles = ['Todos']
    
    segmento_seleccionado = st.sidebar.selectbox(
        "📊 Selecciona Segmento:",
        segmentos_disponibles,
        key="segmento"
    )
    
    # Filtrar basado en Segmento
    if segmento_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['segmento'] == segmento_seleccionado]
    
    # Filtro 4: Cliente específico (opcional) - CON ID
    if len(df_filtrado) > 0:
        # Crear opciones que muestren ID + Nombre
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
        
        # Filtrar basado en Cliente seleccionado
        if cliente_seleccionado != 'Todos':
            if ' - ' in cliente_seleccionado:
                id_seleccionado = cliente_seleccionado.split(' - ')[0]
                df_filtrado = df_filtrado[df_filtrado['id_cliente'] == id_seleccionado]
            else:
                df_filtrado = df_filtrado[df_filtrado['nombre_cliente'] == cliente_seleccionado]
    
    return df_filtrado

# -------------------------
# ANÁLISIS DE DATOS FILTRADOS - CON ID
# -------------------------
def show_filtered_analysis(df_filtrado, df_original):
    """Muestra análisis basado en los filtros seleccionados"""
    
    st.header("📈 Análisis de Datos Filtrados")
    
    # Mostrar información del filtro actual
    st.write(f"**Mostrando:** {len(df_filtrado)} de {len(df_original)} clientes")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_clientes = len(df_filtrado)
        st.metric("👥 Clientes", total_clientes)
    
    with col2:
        total_cap_2024 = df_filtrado['cap_2024'].sum() if 'cap_2024' in df_filtrado.columns and not df_filtrado['cap_2024'].isna().all() else 0
        st.metric("💰 CAP 2024", f"${total_cap_2024:,.0f}")
    
    with col3:
        total_cap_2025 = df_filtrado['cap_2025'].sum() if 'cap_2025' in df_filtrado.columns and not df_filtrado['cap_2025'].isna().all() else 0
        st.metric("💰 CAP 2025", f"${total_cap_2025:,.0f}")
    
    with col4:
        if total_cap_2024 > 0:
            crecimiento_total = (total_cap_2025 - total_cap_2024) / total_cap_2024
        else:
            crecimiento_total = 0
        st.metric("📈 Crecimiento Total", f"{crecimiento_total:.2%}")
    
    # Mostrar datos filtrados - CON ID_CLIENTE
    st.subheader("📋 Datos Filtrados")
    
    # Seleccionar columnas para mostrar - INCLUYENDO ID_CLIENTE
    columnas_a_mostrar = ['id_cliente', 'sucursal', 'oficina', 'segmento', 'nombre_cliente', 
                         'cap_2024', 'cap_2025', 'diferencia', 'crec']
    
    # Agregar columnas adicionales si existen
    columnas_adicionales = ['colocacion', 'recaudo', 'nomina', 'margen']
    for col in columnas_adicionales:
        if col in df_filtrado.columns and not df_filtrado[col].isna().all():
            columnas_a_mostrar.append(col)
    
    df_mostrar = df_filtrado[columnas_a_mostrar].copy()
    
    # Formatear números para mejor visualización
    for col in df_mostrar.select_dtypes(include=[np.number]).columns:
        if col not in ['id_cliente', 'crec']:  # No formatear ID ni porcentajes
            df_mostrar[col] = df_mostrar[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and abs(x) >= 1000 else f"{x:.2f}" if pd.notna(x) else "N/A")
        elif col == 'crec':
            df_mostrar[col] = df_mostrar[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    
    st.dataframe(df_mostrar, use_container_width=True, height=400)
    
    # Análisis gráfico si hay múltiples registros
    if len(df_filtrado) > 1:
        show_multiple_analysis(df_filtrado)
    elif len(df_filtrado) == 1:
        show_single_client_analysis(df_filtrado.iloc[0])

def show_multiple_analysis(df_filtrado):
    """Muestra análisis cuando hay múltiples clientes filtrados"""
    
    st.subheader("📊 Análisis Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de CAP por sucursal
        cap_por_sucursal = df_filtrado.groupby('sucursal')[['cap_2024', 'cap_2025']].sum().reset_index()
        fig = px.bar(cap_por_sucursal, x='sucursal', y=['cap_2024', 'cap_2025'],
                    title="CAP 2024 vs 2025 por Sucursal", 
                    barmode='group',
                    labels={'value': 'CAP', 'variable': 'Año'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de crecimiento por segmento
        crecimiento_segmento = df_filtrado.groupby('segmento').agg({
            'crec': 'mean',
            'id_cliente': 'count'
        }).reset_index()
        
        fig = px.pie(crecimiento_segmento, values='id_cliente', names='segmento',
                    title="Distribución de Clientes por Segmento",
                    hover_data=['crec'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 clientes por CAP 2025 - CON ID
    st.subheader("🏆 Top 10 Clientes por CAP 2025")
    top_clientes = df_filtrado.nlargest(10, 'cap_2025')[['id_cliente', 'nombre_cliente', 'cap_2024', 'cap_2025', 'crec']]
    
    # Crear etiquetas con ID + Nombre para el gráfico
    top_clientes['etiqueta'] = top_clientes['id_cliente'] + ' - ' + top_clientes['nombre_cliente'].str.slice(0, 30)
    
    fig = px.bar(top_clientes, x='etiqueta', y='cap_2025',
                title="Top 10 Clientes - CAP 2025",
                hover_data=['cap_2024', 'crec'],
                labels={'etiqueta': 'Cliente (ID - Nombre)', 'cap_2025': 'CAP 2025'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def show_single_client_analysis(cliente):
    """Muestra análisis detallado para un solo cliente - CON ID"""
    
    st.subheader(f"👤 Análisis Detallado: {cliente['id_cliente']} - {cliente['nombre_cliente']}")
    
    # Mostrar información básica del cliente
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write(f"**ID Cliente:** {cliente['id_cliente']}")
        st.write(f"**Sucursal:** {cliente['sucursal']}")
        st.write(f"**Oficina:** {cliente['oficina']}")
    
    with col_info2:
        st.write(f"**Segmento:** {cliente['segmento']}")
        st.write(f"**Nombre:** {cliente['nombre_cliente']}")
    
    # Métricas financieras
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CAP 2024", 
                 f"${cliente['cap_2024']:,.0f}" if pd.notna(cliente['cap_2024']) else "N/A",
                 delta=f"{cliente['crec']:.2%}" if pd.notna(cliente['crec']) else "N/A")
        st.metric("CAP 2025", 
                 f"${cliente['cap_2025']:,.0f}" if pd.notna(cliente['cap_2025']) else "N/A")
    
    with col2:
        st.metric("Diferencia", 
                 f"${cliente['diferencia']:,.0f}" if pd.notna(cliente['diferencia']) else "N/A")
        if pd.notna(cliente['colocacion']):
            st.metric("Colocación", f"${cliente['colocacion']:,.0f}")
    
    with col3:
        if pd.notna(cliente['recaudo']):
            st.metric("Recaudo", f"${cliente['recaudo']:,.0f}")
        if pd.notna(cliente['margen']):
            st.metric("Margen", f"${cliente['margen']:,.0f}")
    
    # Gráfico de evolución CAP
    if pd.notna(cliente['cap_2024']) and pd.notna(cliente['cap_2025']):
        evolucion_data = pd.DataFrame({
            'Año': ['2024', '2025'],
            'CAP': [cliente['cap_2024'], cliente['cap_2025']]
        })
        fig = px.line(evolucion_data, x='Año', y='CAP', 
                     title="Evolución del CAP 2024-2025", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# ESTADÍSTICAS GENERALES - CON DATOS NORMALIZADOS
# -------------------------
def show_general_stats(df):
    """Muestra estadísticas generales del dataset"""
    
    st.header("📊 Estadísticas Generales del Dataset")
    
    # Mostrar que los datos están normalizados
    st.success("✅ Datos normalizados: 'cesar' corregido a 'Cesar'")
    st.subheader(f"✅ Total de Clientes: {len(df)}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏢 Distribución por Sucursal")
        sucursal_stats = df['sucursal'].value_counts()
        for sucursal, count in sucursal_stats.items():
            st.write(f"**{sucursal}:** {count} clientes")
    
    with col2:
        st.subheader("📈 Por Segmento")
        segmento_stats = df['segmento'].value_counts()
        for segmento, count in segmento_stats.items():
            st.write(f"**{segmento}:** {count} clientes")
        
        st.subheader("🎯 Crecimiento por Segmento")
        crecimiento_segmento = df.groupby('segmento')['crec'].mean().sort_values(ascending=False)
        for segmento, crecimiento in crecimiento_segmento.items():
            st.write(f"**{segmento}:** {crecimiento:.2%}")
    
    with col3:
        st.subheader("💰 Métricas Globales")
        st.metric("Total Clientes", len(df))
        st.metric("CAP 2024 Total", f"${df['cap_2024'].sum():,.0f}")
        st.metric("CAP 2025 Total", f"${df['cap_2025'].sum():,.0f}")
        st.metric("Crecimiento Promedio", f"{df['crec'].mean():.2%}")
        
        # Cliente con mayor crecimiento
        mayor_crecimiento = df.loc[df['crec'].idxmax()] if not df['crec'].isna().all() else None
        if mayor_crecimiento is not None:
            st.write(f"**Mayor crecimiento:**")
            st.write(f"{mayor_crecimiento['id_cliente']} - {mayor_crecimiento['nombre_cliente']}")
            st.write(f"({mayor_crecimiento['crec']:.2%})")

# -------------------------
# PANTALLA PRINCIPAL
# -------------------------
def main():
    st.title("🏦 Análisis Financiero - Datos Normalizados")
    st.markdown("Selecciona filtros en el panel lateral para explorar los datos de clientes")
    
    # Cargar datos
    df = load_and_normalize_data()
    
    if df is None:
        return
    
    # Mostrar información general del dataset en el sidebar
    st.sidebar.header("📊 Información del Dataset")
    st.sidebar.write(f"**Clientes totales:** {len(df)}")
    st.sidebar.write(f"**Sucursales:** {len(df['sucursal'].unique())}")
    st.sidebar.write(f"**Oficinas:** {len(df['oficina'].unique())}")
    st.sidebar.write(f"**Segmentos:** {len(df['segmento'].unique())}")
    
    # Mostrar las sucursales detectadas
    st.sidebar.write("**Sucursales encontradas:**")
    for sucursal in sorted(df['sucursal'].unique()):
        st.sidebar.write(f"- {sucursal}")
    
    # Configurar sistema de filtros
    df_filtrado = setup_filters(df)
    
    # Mostrar análisis basado en filtros
    show_filtered_analysis(df_filtrado, df)
    
    # Mostrar estadísticas generales
    show_general_stats(df)

if __name__ == "__main__":
    main()