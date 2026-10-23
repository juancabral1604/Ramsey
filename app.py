import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ramsey_core import (
    RamseyParameters,
    classify_trajectory,
    compute_backward_stable_path,
    compute_bvp_path,
    compute_linear_analysis,
    compute_local_consumption,
    compute_steady_state,
    production,
    shadow_price_path,
    simulate_ramsey,
    validate_parameters,
)

st.set_page_config(
    page_title="Modelo de Ramsey-Cass-Koopmans",
    page_icon="📈",
    layout="wide"
)

st.title("Explorador del modelo de Ramsey-Cass-Koopmans")
st.caption("Dinámica de capital y consumo, estabilidad de punto silla y sensibilidad a las condiciones iniciales")


with st.sidebar:
    st.header("Parámetros")
    A = st.number_input(
        "Productividad A",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="El máximo de A es un límite práctico del aplicativo, no una frontera teórica del modelo."
    )
    alpha = st.number_input(
        "Elasticidad del capital α",
        min_value=0.01,
        max_value=0.99,
        value=0.33,
        step=0.01
    )
    rho = st.number_input(
        "Impaciencia ρ",
        min_value=0.001,
        max_value=1.0,
        value=0.04,
        step=0.005,
        format="%.3f",
        help="El límite superior es una restricción práctica de la interfaz, no una frontera teórica del modelo."
    )
    theta = st.number_input(
        "Inversa de la EIS θ",
        min_value=0.10,
        max_value=20.0,
        value=2.0,
        step=0.10,
        help="El máximo de θ es un límite práctico del aplicativo, no una frontera teórica del modelo."
    )
    delta = st.number_input(
        "Depreciación δ",
        min_value=0.0,
        max_value=0.99,
        value=0.05,
        step=0.01
    )
    n = st.number_input(
        "Crecimiento poblacional n",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.005,
        format="%.3f",
        help="El límite superior es una restricción práctica de la interfaz, no una frontera teórica del modelo."
    )
    T = st.slider(
        "Horizonte T",
        min_value=10,
        max_value=1000,
        value=100,
        step=5
    )


params = RamseyParameters(A=A, alpha=alpha, rho=rho, theta=theta, delta=delta, n=n)
parameter_errors = validate_parameters(params, T=T)

if parameter_errors:
    for error in parameter_errors:
        st.error(error)
    st.stop()


try:
    steady_state = compute_steady_state(params)

    if steady_state.k_star > 1e10:
        st.warning(
            "La parametrización genera una escala de capital "
            "estacionario extremadamente grande. Algunos métodos "
            "numéricos pueden no converger o producir resultados "
            "poco informativos. Esto es un límite computacional "
            "del aplicativo, no una restricción teórica adicional "
            "sobre α."
        )

    if steady_state.k_star > 1e4:
        st.warning(
            "El capital estacionario es muy grande para esta "
            "parametrización. Aunque el resultado puede ser "
            "matemáticamente válido, las escalas gráficas y los "
            "tiempos numéricos pueden resultar poco informativos."
        )

    linear_analysis = compute_linear_analysis(params, steady_state)

except Exception as error:
    st.error(str(error))
    st.stop()


with st.sidebar:
    st.header("Condiciones iniciales")
    k_0 = st.number_input(
        "Capital inicial k₀",
        min_value=0.0001,
        max_value=float(max(100.0, 5.0 * steady_state.k_star)),
        value=float(0.9 * steady_state.k_star),
        step=float(max(0.001, steady_state.k_star / 100.0)),
        format="%.6f"
    )


c_0_local = compute_local_consumption(
    k_0,
    steady_state,
    linear_analysis
)

bvp = None
bvp_error = None
c_0_bvp = np.nan
bvp_available = False

try:
    bvp = compute_bvp_path(
        k_0,
        params,
        steady_state,
        linear_analysis,
        T=T
    )

    if bvp["success"]:
        c_0_bvp = bvp["c_0_bvp"]
        bvp_available = True
    else:
        bvp_error = bvp["message"]

except Exception as error:
    bvp_error = str(error)


backward = None
backward_error = None
c_0_backward = np.nan

try:
    backward = compute_backward_stable_path(
        k_0,
        params,
        steady_state,
        linear_analysis
    )
    c_0_backward = backward["c_0_backward"]

except Exception as error:
    backward_error = str(error)


if backward is not None:
    c_0_reference = c_0_backward
    stable_method = "Backward adaptativo"
elif bvp_available:
    c_0_reference = c_0_bvp
    stable_method = "BVP de respaldo"
else:
    c_0_reference = c_0_local
    stable_method = "Lineal local"

if not bvp_available:
    st.warning(
        "El método BVP no está disponible. "
        f"Detalle: {bvp_error}"
    )

if backward is None:
    st.warning(
        "La integración backward no está disponible. "
        f"Detalle del error: {backward_error}"
    )


with st.sidebar:
    
    consumption_options = [
        "Ingresar manualmente",
        "Usar aproximación local"
    ]

    if backward is not None:
        consumption_options.insert(1, "Usar aproximación backward")

    if bvp_available:
        consumption_options.append("Usar aproximación BVP")


    consumption_mode = st.selectbox(
        "Valor inicial de consumo",
        consumption_options
    )

    suggested_default = float(max(c_0_reference, 0.0001))

    manual_c_0 = st.number_input(
        "Consumo inicial c₀",
        min_value=0.0001,
        max_value=float(max(100.0, 5.0 * steady_state.c_star)),
        value=suggested_default,
        step=float(max(0.0001, steady_state.c_star / 1000.0)),
        format="%.9f",
        disabled=consumption_mode != "Ingresar manualmente"
    )

if consumption_mode == "Usar aproximación backward":
    c_0 = c_0_backward
elif consumption_mode == "Usar aproximación local":
    c_0 = c_0_local
elif consumption_mode == "Usar aproximación BVP":
    c_0 = c_0_bvp
else:
    c_0 = manual_c_0


try:
    result = simulate_ramsey(k_0, c_0, params, steady_state, T=T)
    diagnosis = classify_trajectory(result, steady_state)
except Exception as error:
    st.error(str(error))
    st.stop()

st.subheader("Resultado principal")
metric_columns = st.columns(6)
metric_columns[0].metric("k*", f"{steady_state.k_star:.6f}")
metric_columns[1].metric("c*", f"{steady_state.c_star:.6f}")
metric_columns[2].metric("c₀ elegido", f"{c_0:.9f}")
metric_columns[3].metric("c₀ local", f"{c_0_local:.9f}")
if backward is not None:
    metric_columns[4].metric("c₀ backward", f"{c_0_backward:.9f}")
else:
    metric_columns[4].metric("c₀ backward", "No disponible")

if bvp_available:
    metric_columns[5].metric("c₀ BVP", f"{c_0_bvp:.9f}")
else:
    metric_columns[5].metric("c₀ BVP", "No disponible")

if diagnosis["classification"] == "Descapitalización":
    st.error(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')
elif diagnosis["classification"] == "Sobreacumulación":
    st.warning(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')
elif diagnosis["classification"] == "Cercanía al equilibrio en el horizonte simulado":
    st.success(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')
else:
    st.info(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')

with st.expander("Diagnóstico técnico de la trayectoria"):
    st.write(
        "Distancia final relativa al equilibrio:",
        f'{diagnosis["relative_final_distance"]:.8%}'
    )
    st.write(
        "¿Se aproxima al equilibrio al final?:",
        diagnosis["approaching_at_end"]
    )
    st.write(
        "Distancia final:",
        f'{result["final_distance"]:.10f}'
    )
    st.write(
        "Distancia mínima:",
        f'{result["minimum_distance"]:.10f}'
    )

st.caption("La cercanía observada en un horizonte finito no constituye una demostración de convergencia cuando t tiende a infinito.")

phase_tab, time_tab, methods_tab, guide_tab = st.tabs([
    "Diagrama de fases",
    "Dinámica temporal",
    "Métodos numéricos",
    "Guía del modelo"
])

with phase_tab:
    
    k_max_data = max(
        steady_state.k_star,
        k_0,
        float(np.nanmax(result["k_path"]))
    )

    if backward is not None:
        k_max_data = max(
            k_max_data,
            float(np.nanmax(backward["k_path"]))
        )
    elif bvp_available:
        k_max_data = max(
            k_max_data,
            float(np.nanmax(bvp["k_path"]))
        )

    k_grid = np.linspace(0.0001, max(1.6 * steady_state.k_star, 1.1 * k_max_data), 700)
    capital_nullcline = production(k_grid, params) - (params.n + params.delta) * k_grid
    capital_nullcline = np.where(capital_nullcline >= 0, capital_nullcline, np.nan)

    phase_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Diagrama completo", "Zoom alrededor del equilibrio")
    )

    for column in [1, 2]:
        phase_figure.add_trace(
            go.Scatter(
                x=k_grid,
                y=capital_nullcline,
                mode="lines",
                name="Nulclina k̇ = 0",
                line=dict(color="black", width=2),
                legendgroup="k-nullcline",
                showlegend=column == 1
            ),
            row=1,
            col=column
        )
        phase_figure.add_vline(
            x=steady_state.k_star,
            line_width=2,
            line_dash="dash",
            line_color="gray",
            row=1,
            col=column
        )
        
        
        if backward is not None:
            stable_k_path = backward["k_path"][::-1]
            stable_c_path = backward["c_path"][::-1]
            stable_path_label = "Senda estable backward"

        elif bvp_available:
            stable_k_path = bvp["k_path"]
            stable_c_path = bvp["c_path"]
            stable_path_label = "Senda estable BVP"

        else:
            stable_k_path = None
            stable_c_path = None
            stable_path_label = None

        if stable_k_path is not None:
            phase_figure.add_trace(
                go.Scatter(
                    x=stable_k_path,
                    y=stable_c_path,
                    mode="lines",
                    name=stable_path_label,
                    line=dict(color="limegreen", width=5),
                    legendgroup="stable-path",
                    showlegend=column == 1
                ),
                row=1,
                col=column
            )


        phase_figure.add_trace(
            go.Scatter(
                x=result["k_path"],
                y=result["c_path"],
                mode="lines",
                name="Trayectoria del c₀ elegido",
                line=dict(color="royalblue", width=3),
                legendgroup="chosen-path",
                showlegend=column == 1,
                customdata=result["time"],
                hovertemplate="k=%{x:.6f}<br>c=%{y:.6f}<br>t=%{customdata:.3f}<extra></extra>"
            ),
            row=1,
            col=column
        )
        phase_figure.add_trace(
            go.Scatter(
                x=[steady_state.k_star],
                y=[steady_state.c_star],
                mode="markers",
                name="Estado estacionario",
                marker=dict(color="black", size=15, symbol="star"),
                legendgroup="steady-state",
                showlegend=column == 1
            ),
            row=1,
            col=column
        )
        phase_figure.add_trace(
            go.Scatter(
                x=[k_0],
                y=[c_0],
                mode="markers",
                name="Condición inicial",
                marker=dict(color="royalblue", size=10, line=dict(color="black", width=1)),
                legendgroup="initial-point",
                showlegend=column == 1
            ),
            row=1,
            col=column
        )

    phase_figure.update_xaxes(
        range=[0, max(1.6 * steady_state.k_star, 1.05 * k_max_data)],
        title_text="Capital por persona, k",
        row=1,
        col=1
    )
    phase_figure.update_yaxes(
        range=[0, max(1.6 * steady_state.c_star, 1.05 * float(np.nanmax(result["c_path"])))],
        title_text="Consumo por persona, c",
        row=1,
        col=1
    )
    phase_figure.update_xaxes(
        range=[min(0.95 * min(k_0, steady_state.k_star), 0.95 * steady_state.k_star), 1.05 * max(k_0, steady_state.k_star)],
        title_text="Capital por persona, k",
        row=1,
        col=2
    )
    
    zoom_consumption_values = [c_0, steady_state.c_star]

    if backward is not None:
        zoom_consumption_values.append(c_0_backward)

    zoom_c_min = max(
        0,
        min(zoom_consumption_values) - 0.05 * steady_state.c_star
    )
    zoom_c_max = (
        max(zoom_consumption_values) + 0.05 * steady_state.c_star
    )

    phase_figure.update_yaxes(
        range=[zoom_c_min, zoom_c_max],
        title_text="Consumo por persona, c",
        row=1,
        col=2
    )
    phase_figure.update_layout(
        height=650,
        title="Diagrama de fases del modelo de Ramsey",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    st.plotly_chart(phase_figure, use_container_width=True)

with time_tab:
    lambda_path = shadow_price_path(result["c_path"], params.theta)
    lambda_star = steady_state.c_star ** (-params.theta)
    time_figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Capital",
            "Consumo",
            "Distancia al estado estacionario",
            "Valor sombra"
        )
    )
    time_figure.add_trace(
        go.Scatter(x=result["time"], y=result["k_path"], mode="lines", name="k(t)", line=dict(color="royalblue", width=3)),
        row=1,
        col=1
    )
    time_figure.add_hline(y=steady_state.k_star, line_dash="dash", line_color="black", row=1, col=1)
    time_figure.add_trace(
        go.Scatter(x=result["time"], y=result["c_path"], mode="lines", name="c(t)", line=dict(color="darkgreen", width=3)),
        row=1,
        col=2
    )
    time_figure.add_hline(y=steady_state.c_star, line_dash="dash", line_color="black", row=1, col=2)
    time_figure.add_trace(
        go.Scatter(x=result["time"], y=result["distance_path"], mode="lines", name="Distancia", line=dict(color="darkorange", width=3)),
        row=2,
        col=1
    )
    time_figure.add_trace(
        go.Scatter(x=result["time"], y=lambda_path, mode="lines", name="λ(t)", line=dict(color="purple", width=3)),
        row=2,
        col=2
    )
    time_figure.add_hline(y=lambda_star, line_dash="dash", line_color="black", row=2, col=2)
    time_figure.update_yaxes(type="log", row=2, col=1)
    time_figure.update_yaxes(type="log", row=2, col=2)
    time_figure.update_xaxes(title_text="Tiempo", row=1, col=1)
    time_figure.update_xaxes(title_text="Tiempo", row=1, col=2)
    time_figure.update_xaxes(title_text="Tiempo", row=2, col=1)
    time_figure.update_xaxes(title_text="Tiempo", row=2, col=2)
    time_figure.update_layout(height=800, title="Evolución temporal", hovermode="x unified")
    st.plotly_chart(time_figure, use_container_width=True)

    summary = pd.DataFrame({
        "Indicador": [
            "Capital final",
            "Consumo final",
            "Distancia mínima",
            "Tiempo de mayor aproximación",
            "Distancia final",
            "Capital agotado"
        ],
        "Valor": [
            result["final_k"],
            result["final_c"],
            result["minimum_distance"],
            result["closest_time"],
            result["final_distance"],
            result["capital_exhausted"]
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

with methods_tab:
    
    methods = [
        ("Lineal local", c_0_local)
    ]

    if backward is not None:
        methods.append(("Integración backward", c_0_backward))

    if bvp_available:
        methods.append(("BVP", c_0_bvp))

    methods_table = pd.DataFrame({
        "Método": [method[0] for method in methods],
        "c₀ aproximado": [method[1] for method in methods],
    })

    if backward is not None:
        methods_table["Diferencia respecto de backward"] = [
            value - c_0_backward for _, value in methods
        ]


    st.dataframe(methods_table, use_container_width=True, hide_index=True)

    method_columns = st.columns(4)
    method_columns[0].metric("det(J*)", f'{linear_analysis["determinant"]:.8f}')
    method_columns[1].metric("Traza(J*)", f'{linear_analysis["trace"]:.8f}')
    method_columns[2].metric("λ estable", f'{linear_analysis["lambda_s"]:.8f}')
    method_columns[3].metric("λ inestable", f'{linear_analysis["lambda_u"]:.8f}')

    st.write("Jacobiano evaluado en el estado estacionario")
    st.dataframe(
        pd.DataFrame(
            linear_analysis["jacobian"],
            index=["k̇", "ċ"],
            columns=["k", "c"]
        ),
        use_container_width=True
    )
    
 
    st.write("Diagnóstico del método BVP")

    if bvp_available:
        st.write("Estado:", bvp["message"])

        st.metric(
            "Residuo máximo de las condiciones de contorno",
            f"{np.max(np.abs(bvp['boundary_residuals'])):.3e}"
        )

        st.caption(
            "Backward y BVP aproximan la senda estable mediante "
            "procedimientos numéricos diferentes. Su cercanía "
            "funciona como control de robustez."
        )
    else:
        st.warning(
            "El método BVP no está disponible. "
            f"Detalle: {bvp_error}"
        )


with guide_tab:
    st.title("Guía del modelo de Ramsey-Cass-Koopmans")

    st.markdown(
        """
        Esta guía explica la estructura económica del modelo, sus ecuaciones,
        el estado estacionario y los métodos numéricos utilizados para estudiar
        la transición de la economía hacia su equilibrio de largo plazo.
        """
    )

    st.divider()

    st.subheader("1. Objetivo y estructura del modelo")

    st.write(
        "El modelo de Ramsey-Cass-Koopmans estudia cómo una economía decide "
        "distribuir sus recursos entre consumo presente y acumulación de capital. "
        "Las decisiones de los hogares tienen en cuenta tanto sus preferencias "
        "intertemporales como el rendimiento de la inversión."
    )

    st.write(
        "El modelo describe la evolución del capital por persona y del consumo "
        "por persona. La economía puede encontrarse fuera de su estado estacionario "
        "y ajustar ambas variables a lo largo del tiempo."
    )

    st.write(
        "El objetivo de la simulación es estudiar las trayectorias dinámicas "
        "y aproximar la senda estable que conduce al estado estacionario."
    )

    st.subheader("2. Ecuaciones fundamentales")

    st.markdown("**Producción por persona**")
    st.latex(r"y=f(k)=Ak^{\alpha}")

    st.write(
        "La producción por persona depende del capital por persona. "
        "A representa la productividad total de los factores y alpha "
        "representa la elasticidad de la producción respecto del capital."
    )

    st.markdown("**Acumulación de capital**")
    st.latex(r"\dot{k}=Ak^{\alpha}-c-(n+\delta)k")

    st.write(
        "La variación del capital por persona es igual a la producción "
        "menos el consumo y la inversión necesaria para compensar la "
        "depreciación del capital y el crecimiento de la población."
    )

    st.markdown("**Evolución óptima del consumo**")
    st.latex(
        r"\frac{\dot{c}}{c}"
        r"=\frac{\alpha A k^{\alpha-1}-\delta-\rho}{\theta}"
    )

    st.write(
        "La tasa de crecimiento del consumo depende de la productividad "
        "marginal del capital, la depreciación, la impaciencia y la inversa "
        "de la elasticidad de sustitución intertemporal."
    )

    st.write(
        "Cuando el rendimiento marginal neto del capital supera la tasa "
        "de descuento, el consumo tiende a crecer. Cuando queda por debajo, "
        "el consumo tiende a disminuir."
    )

    st.write(
        "Un valor mayor de theta reduce la respuesta del crecimiento del "
        "consumo ante una misma diferencia entre el rendimiento marginal "
        "neto del capital y la tasa de descuento."
    )

    st.subheader("3. Estado estacionario")

    st.write(
        "En el estado estacionario, el capital y el consumo por persona "
        "permanecen constantes. Por lo tanto, sus tasas de variación son cero."
    )

    st.markdown("**Capital estacionario**")
    st.latex(
        r"k^*=\left(\frac{\alpha A}{\rho+\delta}\right)"
        r"^{\frac{1}{1-\alpha}}"
    )

    st.markdown("**Consumo estacionario**")
    st.latex(r"c^*=A(k^*)^{\alpha}-(n+\delta)k^*")

    st.write(
        "El capital estacionario se obtiene igualando la productividad "
        "marginal del capital a la suma de la tasa de descuento y la "
        "depreciación."
    )

    st.write(
        "Una vez obtenido el capital estacionario, el consumo estacionario "
        "se calcula a partir de la ecuación de acumulación de capital."
    )

    st.info(
        "En esta formulación, theta afecta la dinámica de ajuste, pero no "
        "los valores del capital y del consumo estacionarios."
    )

    st.subheader("4. Parámetros y variables")

    st.markdown("**Variables de estado y control**")

    st.write(
        "Capital por persona (k): stock de capital disponible por persona. "
        "Es una variable de estado predeterminada cuya evolución depende "
        "de la inversión y de la depreciación efectiva."
    )

    st.write(
        "Consumo por persona (c): consumo individual representativo. "
        "Es la variable de control que determina cuánto de la producción "
        "se utiliza para consumo en lugar de acumular capital."
    )

    
    
    
    st.markdown("**Parámetros económicos**")

    parametros = [
        (r"A", "Productividad total de los factores. Un aumento de A eleva la producción para un mismo nivel de capital."),
        (r"\alpha", "Elasticidad del producto respecto del capital. Determina la contribución del capital a la producción y la forma de sus rendimientos marginales."),
        (r"\rho", "Tasa de descuento subjetiva. Un valor mayor representa mayor impaciencia y modifica el rendimiento requerido para mantener constante el consumo."),
        (r"\theta", "Inversa de la elasticidad de sustitución intertemporal. Controla la sensibilidad del crecimiento del consumo ante cambios en el rendimiento neto del capital."),
        (r"\delta", "Tasa de depreciación del capital. Una parte del capital existente deja de estar disponible en cada período."),
        (r"n", "Tasa de crecimiento de la población. En términos por persona, el crecimiento poblacional diluye el capital existente."),
        (r"T", "Horizonte temporal de la simulación. Determina la extensión del período que se estudia."),
    ]

    for simbolo, descripcion in parametros:
        col_simbolo, col_descripcion = st.columns([1, 10], vertical_alignment="center")
        with col_simbolo:
            st.latex(simbolo)
        with col_descripcion:
            st.write(descripcion)



    st.subheader("5. Supuestos y dominios económicos")

    st.write(
        "La función de producción presenta rendimientos marginales "
        "decrecientes del capital cuando alpha se encuentra entre cero y uno."
    )

    st.latex(r"0<\alpha<1")

    st.write(
        "La productividad debe ser positiva. La tasa de descuento debe "
        "ser positiva y theta también debe ser mayor que cero."
    )

    st.latex(r"A>0,\qquad \rho>0,\qquad \theta>0")

    st.write(
        "La depreciación y el crecimiento poblacional se consideran "
        "no negativos en las parametrizaciones admitidas por la aplicación."
    )

    st.latex(r"\delta\geq 0,\qquad n\geq 0")

    st.write(
        "Estos supuestos definen el dominio económico utilizado por la "
        "aplicación. Los límites adicionales de los controles de entrada "
        "son restricciones prácticas del software y no necesariamente "
        "fronteras teóricas del modelo."
    )

    st.subheader("6. Dinámica y estabilidad de punto silla")

    st.write(
        "El sistema es dinámico: las decisiones actuales condicionan "
        "la trayectoria futura del capital y del consumo."
    )

    st.write(
        "En el estado estacionario, el sistema puede analizarse mediante "
        "una aproximación lineal. La matriz jacobiana describe cómo "
        "pequeñas desviaciones respecto del equilibrio afectan la evolución "
        "de las variables."
    )

    st.write(
        "En el caso estándar de estabilidad de punto silla, existe una "
        "dirección estable y otra inestable. Para converger al equilibrio "
        "de largo plazo, el capital inicial y el consumo inicial deben "
        "ser compatibles con la senda estable."
    )

    st.write(
        "El capital inicial está predeterminado. El consumo inicial, "
        "en cambio, debe ajustarse de manera que la trayectoria satisfaga "
        "la condición de convergencia."
    )

    
    st.subheader("7. Métodos numéricos")

    st.markdown("**7.1. Integración backward**")

    st.write(
        "La integración backward aproxima la senda estable partiendo de un "
        "punto cercano al estado estacionario y resolviendo el sistema hacia "
        "atrás en el tiempo. La idea es aprovechar la dirección estable de "
        "la dinámica local para construir una trayectoria que se aproxime "
        "al equilibrio."
    )

    st.write(
        "Primero se calcula la matriz jacobiana del sistema en el estado "
        "estacionario. Sus autovalores y autovectores permiten identificar "
        "la dirección estable. A partir de ella se construye una condición "
        "inicial próxima al equilibrio y se integra el sistema hacia atrás "
        "para aproximar la senda estable no lineal."
    )

    st.latex(
        r"\begin{pmatrix}"
        r"\dot{k}\\"
        r"\dot{c}"
        r"\end{pmatrix}"
        r"="
        r"\begin{pmatrix}"
        r"Ak^{\alpha}-c-(n+\delta)k\\"
        r"\dfrac{c}{\theta}"
        r"\left(\alpha A k^{\alpha-1}-\delta-\rho\right)"
        r"\end{pmatrix}"
    )

    st.write(
        "La integración hacia atrás es una estrategia numérica para "
        "aproximar la variedad estable. No implica que la economía real "
        "evolucione hacia atrás en el tiempo. La trayectoria obtenida "
        "depende de la precisión de la linealización inicial, del tamaño "
        "de la perturbación y del integrador utilizado."
    )

    st.markdown("**7.2. Problema de contorno (BVP)**")

    st.write(
        "El método BVP, por sus siglas en inglés, resuelve un problema de "
        "valores de contorno. En lugar de especificar únicamente las "
        "condiciones iniciales, busca una trayectoria que satisfaga el "
        "sistema diferencial y las condiciones impuestas en los extremos "
        "del intervalo temporal."
    )

    st.write(
        "Para aproximar la senda estable, se discretiza el intervalo de "
        "tiempo y se busca una solución compatible con las condiciones "
        "de contorno. El algoritmo ajusta la trayectoria hasta reducir "
        "el residuo del sistema y satisfacer las restricciones numéricas "
        "establecidas."
    )

    st.latex(
        r"\dot{k}(t)=Ak(t)^{\alpha}-c(t)-(n+\delta)k(t)"
    )

    st.latex(
        r"\dot{c}(t)=\frac{c(t)}{\theta}"
        r"\left(\alpha A k(t)^{\alpha-1}-\delta-\rho\right)"
    )

    st.write(
        "La ventaja del BVP es que utiliza información de ambos extremos "
        "del intervalo. Sin embargo, su convergencia depende de las "
        "condiciones de contorno, de la aproximación inicial, de la malla "
        "temporal y de la escala de las variables."
    )

    st.markdown("**7.3. Simulación forward**")

    st.write(
        "La integración forward parte de un estado inicial dado y resuelve "
        "el sistema hacia adelante en el tiempo. En esta aplicación, se "
        "utiliza para analizar la trayectoria que resulta de los valores "
        "iniciales del capital y del consumo."
    )

    st.latex(r"k(0)=k_0,\qquad c(0)=c_0")

    st.write(
        "Si el consumo inicial es compatible con la senda estable, la "
        "trayectoria puede converger al estado estacionario. Si el consumo "
        "inicial se desvía de ese valor, la dinámica puede alejarse del "
        "equilibrio debido a la dirección inestable del sistema."
    )

    st.write(
        "Este método permite comparar una trayectoria generada a partir "
        "de condiciones iniciales concretas con las sendas estables "
        "aproximadas mediante backward o BVP. La divergencia de una "
        "trayectoria no demuestra por sí sola un error del modelo: también "
        "puede reflejar que el consumo inicial no satisface la condición "
        "de convergencia."
    )

    
    st.markdown("**7.4. Aproximación local mediante la jacobiana**")

    st.write(
        "La aproximación local consiste en linealizar el sistema alrededor "
        "del estado estacionario. Para ello, se expresan las variables como "
        "desviaciones respecto del equilibrio y se utiliza la matriz "
        "jacobiana para aproximar su evolución."
    )

    st.latex(
        r"\begin{pmatrix}"
        r"\dot{k}\\"
        r"\dot{c}"
        r"\end{pmatrix}"
        r"\approx"
    )

    st.latex(
        r"J(k^*,c^*)"
        r"\begin{pmatrix}"
        r"k-k^*\\"
        r"c-c^*"
        r"\end{pmatrix}"
    )

    st.write(
        "La matriz jacobiana J(k*, c*) contiene las derivadas parciales "
        "de las ecuaciones dinámicas, evaluadas en el estado estacionario."
    )

    st.latex(
        r"J(k^*,c^*)="
        r"\begin{pmatrix}"
        r"\dfrac{\partial \dot{k}}{\partial k}"
        r"&"
        r"\dfrac{\partial \dot{k}}{\partial c}\\[6pt]"
        r"\dfrac{\partial \dot{c}}{\partial k}"
        r"&"
        r"\dfrac{\partial \dot{c}}{\partial c}"
        r"\end{pmatrix}_{(k^*,c^*)}"
    )

    st.write(
        "Los autovalores de la matriz jacobiana indican cómo evolucionan "
        "las desviaciones en las direcciones propias del sistema. En el "
        "caso de punto silla, un autovalor es negativo y el otro positivo."
    )

    st.write(
        "La dirección asociada al autovalor negativo es estable: las "
        "desviaciones pequeñas tienden a reducirse a lo largo de ella. "
        "La dirección asociada al autovalor positivo es inestable: las "
        "desviaciones tienden a amplificarse."
    )

    st.write(
        "Esta aproximación permite identificar la dirección estable cerca "
        "del equilibrio y construir una condición inicial para los métodos "
        "numéricos. Sin embargo, es una aproximación local y no representa "
        "exactamente la dinámica no lineal cuando la trayectoria se aleja "
        "del estado estacionario."
    )


    st.markdown("**7.5. Comparación y disponibilidad de los métodos**")

    st.write(
        "Cada método cumple una función diferente: la aproximación local "
        "estudia la dinámica cerca del equilibrio; backward utiliza la "
        "dirección estable local para construir una trayectoria; BVP busca "
        "una solución que satisfaga condiciones de contorno; y forward "
        "simula la evolución a partir de condiciones iniciales elegidas."
    )

    st.write(
        "Los resultados deben compararse teniendo en cuenta las condiciones "
        "iniciales y de contorno, el horizonte temporal, la precisión "
        "numérica y los errores residuales. Las pequeñas diferencias entre "
        "métodos pueden deberse a aproximaciones numéricas y no "
        "necesariamente a inconsistencias económicas."
    )

    st.warning(
        "Los algoritmos pueden fallar o perder precisión ante escalas "
        "extremas, trayectorias muy sensibles o condiciones iniciales "
        "inadecuadas. Por eso, la aplicación intenta ejecutar los métodos "
        "de manera independiente y muestra los resultados disponibles "
        "sin asumir que todos convergerán."
    )


    st.markdown("**Aproximación local**")

    st.write(
        "La aproximación local utiliza la linealización del sistema cerca "
        "del estado estacionario. Es útil para estudiar el comportamiento "
        "local, pero no debe interpretarse como una representación exacta "
        "de toda la trayectoria no lineal."
    )

    st.write(
        "Los métodos numéricos pueden fallar por problemas de convergencia, "
        "escalas extremas o sensibilidad de las ecuaciones. Por eso, la "
        "aplicación intenta resolver los métodos de manera independiente "
        "y utiliza los resultados que estén disponibles."
    )

    st.subheader("8. Cómo interpretar la simulación")

    st.write(
        "El capital inicial indica el stock de capital por persona al "
        "comienzo del período. El consumo inicial determina la trayectoria "
        "que seguirá el sistema desde ese punto."
    )

    st.write(
        "Si el consumo inicial se encuentra sobre la senda estable, "
        "la trayectoria puede converger al estado estacionario. Si se "
        "encuentra fuera de ella, la economía puede alejarse del equilibrio "
        "o mostrar una trayectoria que no satisface la condición de "
        "convergencia deseada."
    )

    st.write(
        "Las diferencias entre métodos deben interpretarse considerando "
        "sus condiciones de contorno, el horizonte temporal y los errores "
        "numéricos. Una diferencia entre trayectorias no implica por sí sola "
        "que el modelo económico sea incorrecto."
    )

    st.write(
        "Los resultados deben evaluarse junto con las métricas de error, "
        "la disponibilidad de cada método y los gráficos de las trayectorias."
    )

    st.subheader("9. Escalas extremas y limitaciones numéricas")

    st.write(
        "Algunas combinaciones de parámetros pueden generar valores muy "
        "grandes de capital estacionario. Aunque esos valores puedan ser "
        "matemáticamente válidos, pueden dificultar la integración numérica "
        "y hacer que los gráficos pierdan utilidad interpretativa."
    )

    st.write(
        "La aplicación utiliza advertencias y límites computacionales "
        "para identificar algunas de estas situaciones. Dichos límites "
        "responden al funcionamiento numérico del aplicativo y no "
        "constituyen nuevas restricciones teóricas del modelo."
    )

    st.warning(
        "En horizontes extensos, errores numéricos pequeños pueden "
        "amplificarse por la presencia de una dirección inestable. "
    )
