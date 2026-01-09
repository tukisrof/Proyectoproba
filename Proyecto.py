import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configuración de la página
st.set_page_config(page_title="Simulador TLC", layout="wide")
st.title(" Toolkit de Muestreo y Teorema del Límite Central")

# --- Barra Lateral: Controles (Componentes Mínimos) ---
st.sidebar.header("Parámetros de Simulación")

dist_name = st.sidebar.selectbox(
    "1. Selector de Distribución Base",
    ["Uniforme", "Exponencial", "Bernoulli"]
)

n = st.sidebar.slider("2. Tamaño muestral (n)", min_value=1, max_value=100, value=30)
N = st.sidebar.number_input("3. Número de repeticiones (N)", min_value=100, max_value=10000, value=2000)

# --- Lógica de Generación de Datos ---
def generate_data(dist, n, N):
    if dist == "Uniforme":
        data = np.random.uniform(0, 1, (N, n))
        mu_teorica, var_teorica = 0.5, 1/12
    elif dist == "Exponencial":
        data = np.random.exponential(1, (N, n))
        mu_teorica, var_teorica = 1.0, 1.0
    else: # Bernoulli
        p = 0.5
        data = np.random.binomial(1, p, (N, n))
        mu_teorica, var_teorica = p, p*(1-p)
    
    # Calcular N medias muestrales
    means = np.mean(data, axis=1)
    return means, mu_teorica, var_teorica

means, mu_t, var_t = generate_data(dist_name, n, N)

# --- Visualización (Histograma y Normal) ---
st.subheader(f"Distribución de las Medias Muestrales ($\\bar{{X}}$)")

fig, ax = plt.subplots(figsize=(10, 5))
count, bins, ignored = ax.hist(means, bins=40, density=True, alpha=0.6, color='#3498db', label="Histograma de $\\bar{X}$")

# Comparación con Normal: N(mu, var/n)
sigma_t = np.sqrt(var_t / n)
x = np.linspace(min(bins), max(bins), 100)
pdf = stats.norm.pdf(x, mu_t, sigma_t)
ax.plot(x, pdf, 'r-', lw=2, label=f'Normal Teórica: $\\mathcal{{N}}({mu_t}, {var_t/n:.4f})$')

ax.set_title(f"Simulación Monte Carlo con {dist_name} (n={n}, N={N})")
ax.legend()
st.pyplot(fig)

# --- Resumen Numérico ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.markdown("###  Resultados de la Simulación")
    st.write(f"Media de $\\bar{{X}}$: `{np.mean(means):.4f}`")
    st.write(f"Varianza de $\\bar{{X}}$: `{np.var(means):.4f}`")

with col2:
    st.markdown("###  Valores Teóricos (TLC)")
    st.write(f"$\mu$ (esperada): `{mu_t:.4f}`")
    st.write(f"$\sigma^2/n$ (varianza esperada): `{var_t/n:.4f}`")

# --- (Opcional) Intervalo de Confianza ---
st.info(f"El Error Estándar (SE) actual es: {sigma_t:.4f}")
# --- (Opcional) Intervalos de Confianza y Cobertura ---
st.divider()
st.subheader("4. Cobertura Empírica de Intervalos de Confianza (95%)")

# 1. Definir el nivel de confianza (95%)
confianza = 0.95
z_score = stats.norm.ppf((1 + confianza) / 2) # Esto es aprox 1.96

# 2. Calcular el Error Estándar Teórico
error_estandar = np.sqrt(var_t / n)

# 3. Calcular límites inferior y superior para TODAS las simulaciones
lim_inferior = means - z_score * error_estandar
lim_superior = means + z_score * error_estandar

# 4. Verificar si la media teórica (mu_t) cayó dentro del intervalo
aciertos = (lim_inferior <= mu_t) & (lim_superior >= mu_t)
cobertura_real = np.mean(aciertos)

# 5. Mostrar métricas
col_a, col_b = st.columns(2)
col_a.metric("Cobertura Esperada", f"{confianza*100:.0f}%")
col_b.metric("Cobertura Real (Empírica)", f"{cobertura_real*100:.2f}%")

if cobertura_real < 0.93:
    st.warning(f" La cobertura es baja ({cobertura_real*100:.2f}%). Esto pasa cuando 'n' es pequeño y la distribución no es Normal. ¡Aumenta n!")
else:
    st.success(f" ¡Excelente! La cobertura real se acerca al 95%. El TLC está funcionando.")

# 6. Visualización de los primeros 50 intervalos
st.write("Visualización de los primeros 50 intervalos (Verde = Contiene a $\mu$, Rojo = No contiene)")
fig2, ax2 = plt.subplots(figsize=(10, 4))

# Solo graficamos los primeros 50 para que se vea bien
n_view = 50
for i in range(n_view):
    color = 'g' if aciertos[i] else 'r'
    ax2.plot([i, i], [lim_inferior[i], lim_superior[i]], color=color, alpha=0.5)
    ax2.plot(i, means[i], 'o', color='blue', markersize=3)

ax2.axhline(mu_t, color='black', linestyle='--', label=f'Media Verdadera ($\mu={mu_t}$)')
ax2.set_title(f"Primeros {n_view} Intervalos de Confianza")
ax2.legend()
st.pyplot(fig2)