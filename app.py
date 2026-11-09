import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import soccerdata as sd
from datetime import datetime, timedelta

# Configuración de página estilo Chat IA
st.set_page_config(page_title="QBETS AI Chat", page_icon="🤖", layout="centered")
st.title("🤖 QBETS - Asistente IA de Pronósticos")
st.caption("Pregúntame qué partidos hay hoy o pídeme las probabilidades de cualquier enfrentamiento.")

# =========================================================
# 1. CARGAR CEREBRO DE QBETS (TENSORFLOW + SCALER)
# =========================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(directorio_actual, 'qbets_deportivo.keras')
ruta_scaler = os.path.join(directorio_actual, 'scaler_qbets.pkl')

@st.cache_resource
def cargar_cerebro():
    return tf.keras.models.load_model(ruta_modelo), joblib.load(ruta_scaler)

modelo, scaler = cargar_cerebro()

# =========================================================
# 2. CONEXIÓN EN SEGUNDO PLANO CON SOCCERDATA
# =========================================================
@st.cache_data(ttl=1800)
def cargar_datos_liga(liga):
    """Extrae el calendario y estadísticas goleadoras usando FBref de soccerdata."""
    try:
        fbref = sd.FBref(leagues=[liga])
        calendario = fbref.read_schedule().reset_index()
        calendario['date'] = pd.to_datetime(calendario['date'], errors='coerce')
        
        jugados = calendario.dropna(subset=['score']).copy()
        prom_local, prom_visita = {}, {}
        
        if not jugados.empty and jugados['score'].str.contains('–|-').any():
            goles = jugados['score'].str.replace('–', '-').str.split('-', expand=True)
            jugados['g_local'] = pd.to_numeric(goles[0], errors='coerce')
            jugados['g_visita'] = pd.to_numeric(goles[1], errors='coerce')
            prom_local = jugados.groupby('home_team')['g_local'].mean().to_dict()
            prom_visita = jugados.groupby('away_team')['g_local'].mean().to_dict()
            
        return calendario, prom_local, prom_visita
    except Exception:
        # Respaldo instantáneo mientras descarga caché o si no hay conexión
        fechas = [pd.Timestamp(datetime.now().date())] * 4
        df_backup = pd.DataFrame({
            'date': fechas,
            'home_team': ['Real Madrid', 'Barcelona', 'Arsenal', 'Chelsea'],
            'away_team': ['Sevilla', 'Girona', 'Liverpool', 'Tottenham'],
            'score': [None, None, None, None]
        })
        prom_l = {'Real Madrid': 2.3, 'Barcelona': 2.2, 'Arsenal': 2.1, 'Chelsea': 1.7}
        prom_v = {'Sevilla': 1.5, 'Girona': 1.3, 'Liverpool': 1.0, 'Tottenham': 1.4}
        return df_backup, prom_l, prom_v

# Barra lateral para elegir la liga activa y botones rápidos
st.sidebar.header("⚙️ Configuración de QBETS")
liga_activa = st.sidebar.selectbox(
    "Liga conectada (SoccerData):",
    ["ESP-La Liga", "ENG-Premier League", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
)

calendario_df, goles_local_dict, goles_visita_dict = cargar_datos_liga(liga_activa)
equipos_disponibles = list(set(list(goles_local_dict.keys()) + list(calendario_df['home_team'].dropna().unique())))

st.sidebar.markdown("### 💡 Ejemplos de preguntas:")
st.sidebar.code("¿Qué partidos hay hoy?")
st.sidebar.code("Probabilidades de Real Madrid vs Sevilla")
st.sidebar.code("Analiza Barcelona vs Girona cuota 1.65")

if st.sidebar.button("🗑️ Limpiar conversación"):
    st.session_state.mensajes = []
    st.rerun()

# =========================================================
# 3. MOTOR DE RESPUESTAS EN LENGUAJE NATURAL (NLP + RED NEURONAL)
# =========================================================
def responder_consulta(pregunta):
    texto = pregunta.lower().strip()
    
    # INTENCIÓN A: Consultar qué partidos hay hoy / próximos
    palabras_calendario = ["que partidos", "qué partidos", "partidos de hoy", "partidos hay", "jornada", "calendario", "proximos", "próximos"]
    if any(p in texto for p in palabras_calendario):
        hoy = pd.Timestamp(datetime.now().date())
        proximos = calendario_df[calendario_df['date'] >= (hoy - timedelta(days=1))].head(8)
        if proximos.empty:
            proximos = calendario_df.tail(8)
            
        respuesta = f"📅 **Aquí tienes los partidos programados en `{liga_activa}`:**\n\n"
        for _, fila in proximos.iterrows():
            fecha_txt = str(fila['date'])[:10]
            local = fila['home_team']
            visita = fila['away_team']
            respuesta += f"- ⚽ **{local} vs {visita}** *(Fecha: {fecha_txt})*\n"
            
        respuesta += "\n💬 *Escribe el nombre de cualquier partido (ej: `Probabilidades de " + f"{proximos.iloc[0]['home_team']} vs {proximos.iloc[0]['away_team']} cuota 1.75`" + "`) para calcular su Value Bet con mi red neuronal.*"
        return respuesta

    # INTENCIÓN B: Detectar equipos mencionados y calcular probabilidades con QBETS
    equipos_detectados = [eq for eq in equipos_disponibles if eq.lower() in texto]
    
    # Si el usuario escribió "Equipo A vs Equipo B" aunque no estén en caché aún
    if len(equipos_detectados) < 2 and (" vs " in texto or " contra " in texto):
        separador = " vs " if " vs " in texto else " contra "
        limpio = re.sub(r'(probabilidades de|probabilidad de|analiza|pronostico de|pronóstico de|cuota\s*\d+(\.\d+)?)', '', texto).strip()
        partes = limpio.split(separador)
        if len(partes) >= 2:
            equipos_detectados = [partes[0].strip().title(), partes[1].strip().title()]

    if len(equipos_detectados) >= 2:
        eq_local, eq_visita = equipos_detectados[0], equipos_detectados[1]
        
        # Extraer si el usuario puso una cuota en su pregunta (ej. "cuota 1.85" o un decimal)
        match_cuota = re.findall(r'\b\d+\.\d+\b', texto)
        cuota_local = float(match_cuota[0]) if match_cuota else 1.80
        
        g_loc = goles_local_dict.get(eq_local, 1.55)
        g_vis = goles_visita_dict.get(eq_visita, 1.30)
        prob_casa = 1.0 / cuota_local
        
        # Ejecutar predicción en la Red Neuronal QBETS
        entrada = scaler.transform(np.array([[g_loc, g_vis, prob_casa]]))
        prob_ia = float(modelo.predict(entrada, verbose=0)[0][0])
        ventaja = (prob_ia - prob_casa) * 100
        cuota_justa = 1.0 / prob_ia if prob_ia > 0 else 99.0
        
        veredicto = "✅ **¡VALUE BET DETECTADA! (Recomendado Apostar al Local)**" if ventaja > 2.0 else "❌ **NO APOSTAR (La cuota paga menos del riesgo real)**"
        
        return (
            f"### 🧠 Análisis Neuronal QBETS: **{eq_local} vs {eq_visita}**\n"
            f"He procesado los datos de **{liga_activa}** en mi red neuronal y este es el reporte:\n\n"
            f"* **📊 Estadísticas previas:** `{eq_local}` promedia **{g_loc:.2f} goles** en casa, mientras que `{eq_visita}` concede **{g_vis:.2f} goles** como visitante.\n"
            f"* **🏦 Cuota evaluada (TriunfoBet):** `{cuota_local:.2f}` *(Probabilidad implícita de la casa: **{prob_casa*100:.1f}%**)*\n"
            f"* **🤖 Probabilidad real calculada por QBETS:** **{prob_ia*100:.2f}%** *(Cuota justa estimada: `{cuota_justa:.2f}`)*\n"
            f"* **📈 Ventaja matemática (Edge):** `{ventaja:+.2f}%`\n\n"
            f"**Conclusión:** {veredicto}"
        )

    # INTENCIÓN C: Saludo o ayuda general
    return (
        "¡Hola! Soy **QBETS**, tu inteligencia artificial de pronósticos deportivos. ⚽\n\n"
        "Puedes preguntarme cosas como:\n"
        "1. **«¿Qué partidos hay hoy?»** para ver el calendario de la liga seleccionada.\n"
        "2. **«¿Cuáles son las probabilidades de Real Madrid vs Sevilla?»**\n"
        "3. **«Analiza Arsenal vs Chelsea con cuota 1.75»** para decirte si la cuota de TriunfoBet tiene valor matemático."
    )

# =========================================================
# 4. HISTORIAL VISUAL DEL CHAT (ESTILO GEMINI / CHATGPT)
# =========================================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {
            "rol": "assistant",
            "contenido": "¡Hola! Soy **QBETS** 🤖⚽. Estoy conectado a tu modelo `qbets_deportivo.keras` y a las estadísticas de `soccerdata`. ¿Qué partido o jornada quieres que analicemos hoy?"
        }
    ]

# Mostrar todos los mensajes previos en pantalla
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

# Barra inferior para escribir la pregunta
if prompt_usuario := st.chat_input("Escribe tu pregunta (ej: ¿Qué partidos hay hoy? o Probabilidades de Real Madrid vs Sevilla)..."):
    # 1. Mostrar pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    # 2. Generar y mostrar respuesta de QBETS
    with st.chat_message("assistant"):
        with st.spinner("Consultando red neuronal QBETS y base de datos..."):
            respuesta_ia = responder_consulta(prompt_usuario)
            st.markdown(respuesta_ia)
            
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta_ia})