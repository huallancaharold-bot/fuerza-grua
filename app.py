import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Simulador de Estabilidad de Grúa", layout="centered")

st.title("Simulador de Estabilidad de Grúa")
st.write("Evaluación de reacciones en estabilizadores considerando rigidez armónica y pluma telescópica.")

@st.cache_data
def cargar_coeficientes():
    try:
        with open("coeficientes_rigidez_orden8.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo 'coeficientes_rigidez_orden8.json'.")
        return None

COEFICIENTES_FOURIER = cargar_coeficientes()

def evaluar_fourier_desde_coefs(theta: float, coefs: list) -> float:
    val = coefs[0]
    orden = (len(coefs) - 1) // 2
    idx = 1
    for n in range(1, orden + 1):
        val += coefs[idx] * np.sin(n * theta) + coefs[idx+1] * np.cos(n * theta)
        idx += 2
    return val

class Aparejo:
    def __init__(self, masa_pasteca: float, masa_eslingas_grilletes: float = 0.0):
        self.masa_pasteca = masa_pasteca
        self.masa_eslingas_grilletes = masa_eslingas_grilletes

    @property
    def masa_total_aparejo(self):
        return self.masa_pasteca + self.masa_eslingas_grilletes

class Carga:
    def __init__(self, masa_util: float, radio: float, theta_deg: float, aparejo: Aparejo = None):
        self.masa_util = masa_util
        self.aparejo = aparejo if aparejo is not None else Aparejo(0.0, 0.0)
        self.masa_total = self.masa_util + self.aparejo.masa_total_aparejo
        self.radio = radio
        self.theta_deg = theta_deg
        self.theta = np.radians(theta_deg)

    @property
    def coordenadas(self):
        x_long = self.radio * np.cos(self.theta)
        y_trans = -self.radio * np.sin(self.theta)
        return x_long, y_trans

class GruaBase:
    def __init__(self, masa_grua: float, masa_cw: float, d_cw_nominal: float, giro_torreta_deg: float):
        self.masa_grua = masa_grua
        self.masa_cw = masa_cw
        self.giro_torreta_deg = giro_torreta_deg
        self.theta_torreta = np.radians(giro_torreta_deg)
        self.d_cw_nominal = d_cw_nominal

    @property
    def pos_grua(self):
        return 0.0, 0.0

    @property
    def pos_cw(self):
        x = -self.d_cw_nominal * np.cos(self.theta_torreta)
        y = self.d_cw_nominal * np.sin(self.theta_torreta)
        return x, y

class ConfigurarEstabilizadores:
    def __init__(self, pads_dict: dict):
        self.pads = pads_dict
        if COEFICIENTES_FOURIER:
            self.k_funcs = {
                '1': lambda theta: evaluar_fourier_desde_coefs(theta, COEFICIENTES_FOURIER["K1"]),
                '2': lambda theta: evaluar_fourier_desde_coefs(theta, COEFICIENTES_FOURIER["K2"]),
                '3': lambda theta: evaluar_fourier_desde_coefs(theta, COEFICIENTES_FOURIER["K3"]),
                '4': lambda theta: evaluar_fourier_desde_coefs(theta, COEFICIENTES_FOURIER["K4"])
            }

    def obtener_parametros(self, theta: float):
        x_long = np.array([p[0] for p in self.pads.values()])
        y_trans = np.array([p[1] for p in self.pads.values()])
        k_dict = {name: func(theta) for name, func in self.k_funcs.items()}
        k_vals = np.array([k_dict[name] for name in self.pads.keys()])
        return x_long, y_trans, k_vals, k_dict

class AnalisisEstabilidadSolver:
    def __init__(self, grua: GruaBase, carga: Carga, estabilizadores: ConfigurarEstabilizadores, longitud_pluma: float = 22.6):
        self.grua = grua
        self.carga = carga
        self.estabilizadores = estabilizadores
        self.g = 9.81
        self.longitud_pluma = longitud_pluma
        
        area_acero = 2.4 * 0.012
        self.masa_pluma = area_acero * self.longitud_pluma * 7850.0 * 1.30

    def _estimar_efecto_pluma(self):
        ratio_cos = min(max(self.carga.radio / self.longitud_pluma, 0.01), 0.99)
        angulo_pluma = np.arccos(ratio_cos)
        r_cg_pluma_global = (self.longitud_pluma / 2.0)
        radio_cg_proyectado = r_cg_pluma_global * np.cos(angulo_pluma)
        return self.masa_pluma, radio_cg_proyectado

    def resolver_reacciones(self):
        x_L, y_L = self.carga.coordenadas
        x_G, y_G = self.grua.pos_grua
        x_CW, y_CW = self.grua.pos_cw

        m_L = self.carga.masa_total
        m_G = self.grua.masa_grua
        m_CW = self.grua.masa_cw

        m_pluma, r_cg_pluma = self._estimar_efecto_pluma()
        x_pluma = r_cg_pluma * np.cos(self.carga.theta)
        y_pluma = -r_cg_pluma * np.sin(self.carga.theta)

        w_total = (m_L + m_G + m_CW + m_pluma) * self.g
        theta_actual = self.grua.theta_torreta
        X_long, Y_trans, K_rigidez, k_dict_instantaneo = self.estabilizadores.obtener_parametros(theta_actual)

        M_matrix = np.array([
            [np.sum(K_rigidez), np.sum(K_rigidez * X_long), np.sum(K_rigidez * Y_trans)],
            [np.sum(K_rigidez * X_long), np.sum(K_rigidez * X_long**2), np.sum(K_rigidez * X_long * Y_trans)],
            [np.sum(K_rigidez * Y_trans), np.sum(K_rigidez * X_long * Y_trans), np.sum(K_rigidez * Y_trans**2)]
        ])

        rhs_vector = np.array([
            w_total,
            (m_L*x_L + m_G*x_G + m_CW*x_CW + m_pluma*x_pluma)*self.g,
            (m_L*y_L + m_G*y_G + m_CW*y_G + m_pluma*y_pluma)*self.g
        ])

        A, B, C = np.linalg.solve(M_matrix, rhs_vector)

        sum_masas = m_L + m_G + m_CW + m_pluma
        x_cg_sistema = (m_L*x_L + m_G*x_G + m_CW*x_CW + m_pluma*x_pluma) / sum_masas
        y_cg_sistema = (m_L*y_L + m_G*y_G + m_CW*y_G + m_pluma*y_pluma) / sum_masas

        resultados = {}
        for name, (xi, yi) in self.estabilizadores.pads.items():
            ki = k_dict_instantaneo[name]
            Fi_newtons = ki * (A + B*xi + C*yi)
            resultados[name] = Fi_newtons / (self.g * 1000.0)

        return resultados, (x_cg_sistema, y_cg_sistema)

# ==========================================
# INTERFAZ WEB CON STREAMLIT
# ==========================================
st.sidebar.header("Parámetros de Operación")
masa_util = st.sidebar.number_input("Masa Útil (kg)", value=8000.0)
radio = st.sidebar.number_input("Radio de Izaje (m)", value=20.1)
angulo_giro = st.sidebar.number_input("Ángulo de Giro (°)", value=-57.0)
longitud_pluma = st.sidebar.number_input("Longitud de Pluma (m)", value=30.1)

st.sidebar.subheader("Accesorios y Aparejos")
masa_pasteca = st.sidebar.number_input("Masa Pasteca / Gancho (kg)", value=700.0)
masa_eslingas = st.sidebar.number_input("Masa Eslingas y Grilletes (kg)", value=50.0)

st.sidebar.subheader("Configuración de Grúa y Contrapeso")
masa_grua = st.sidebar.number_input("Masa Chasis Grúa (kg)", value=48000.0)
masa_cw = st.sidebar.number_input("Masa Contrapeso (kg)", value=28200.0)
d_cw_nominal = st.sidebar.number_input("Distancia CG Contrapeso (m)", value=4.1)

st.sidebar.subheader("Ubicación de Estabilizadores (Pads)")
p1_x = st.sidebar.number_input("Pad 1 - X (longitudinal)", value=2.8)
p1_y = st.sidebar.number_input("Pad 1 - Y (transversal)", value=-3.5)

p2_x = st.sidebar.number_input("Pad 2 - X (longitudinal)", value=-5.7)
p2_y = st.sidebar.number_input("Pad 2 - Y (transversal)", value=-3.5)

p3_x = st.sidebar.number_input("Pad 3 - X (longitudinal)", value=-5.5)
p3_y = st.sidebar.number_input("Pad 3 - Y (transversal)", value=3.5)

p4_x = st.sidebar.number_input("Pad 4 - X (longitudinal)", value=3.1)
p4_y = st.sidebar.number_input("Pad 4 - Y (transversal)", value=3.5)

if st.button("Ejecutar Análisis y Visualización"):
    if COEFICIENTES_FOURIER is None:
        st.stop()
        
    aparejo_op = Aparejo(masa_pasteca=masa_pasteca, masa_eslingas_grilletes=masa_eslingas)
    carga_op = Carga(masa_util=masa_util, radio=radio, theta_deg=angulo_giro, aparejo=aparejo_op)
    grua_op = GruaBase(masa_grua=masa_grua, masa_cw=masa_cw, d_cw_nominal=d_cw_nominal, giro_torreta_deg=angulo_giro)
    
    apoyos_op = {
        '1': (p1_x, p1_y), '2': (p2_x, p2_y),
        '3': (p3_x, p3_y), '4': (p4_x, p4_y)
    }
    
    estabilizadores_op = ConfigurarEstabilizadores(apoyos_op)
    solver_op = AnalisisEstabilidadSolver(grua=grua_op, carga=carga_op, estabilizadores=estabilizadores_op, longitud_pluma=longitud_pluma)
    
    reacciones, (x_cg, y_cg) = solver_op.resolver_reacciones()
    
    st.subheader("Resultados Numéricos")
    st.write(f"**Centro de Masa Global (Sistema):** X = {x_cg:.2f} m, Y = {y_cg:.2f} m")
    
    for pad, fuerza in reacciones.items():
        if fuerza < 0:
            st.error(f"Pad {pad}: {fuerza:.2f} t — ¡ALERTA: En tensión (Riesgo de volcado)!")
        else:
            st.success(f"Pad {pad}: {fuerza:.2f} t")

    st.subheader("Visualización del Modelo")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    chasis = plt.Rectangle((-1.0, -5.0), 2.0, 10.0, color='blue', alpha=0.20, label="Chasis Grúa")
    ax.add_patch(chasis)

    p_orden = ['2', '3', '4', '1', '2']
    px = [apoyos_op[p][0] for p in p_orden]
    py = [apoyos_op[p][1] for p in p_orden]
    ax.plot(py, px, 'k--', label="Polígono de Sustentación")

    for name, (xi, yi) in apoyos_op.items():
        ax.scatter(yi, xi, color='red', s=120, zorder=5)
        ax.text(yi + 0.4, xi, f"{name}", fontsize=11, fontweight='bold', color='black')

    ax.scatter([0], [0], color='black', marker='X', s=150, zorder=5, label="Centro de Giro")
    x_cw, y_cw = grua_op.pos_cw
    cw_box = plt.Rectangle((y_cw - 0.75, x_cw - 0.5), 1.5, 1.0, color='purple', alpha=0.6, label="Contrapeso")
    ax.add_patch(cw_box)

    x_L, y_L = carga_op.coordenadas
    ax.plot([0, y_L], [0, x_L], color='green', linewidth=2, label="Radio de Izaje")
    ax.scatter([y_L], [x_L], color='white', edgecolor='green', s=200, linewidth=2, zorder=6, label="Carga Total")
    ax.scatter([y_cg], [x_cg], color='orange', marker='h', s=150, zorder=6, label="Centro de Masa")

    ax.axhline(0, color='red', linewidth=1)
    ax.axvline(0, color='red', linewidth=1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-11, 19)
    ax.set_xlabel("Eje Transversal (Y) [m]")
    ax.set_ylabel("Eje Longitudinal (X) [m]")
    ax.set_title(f"Giro: {grua_op.giro_torreta_deg:.1f}° - L_pluma: {longitud_pluma}m", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
