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
    A = st.number_input("Productividad A", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    alpha = st.number_input("Elasticidad del capital α", min_value=0.01, max_value=0.99, value=0.33, step=0.01)
    rho = st.number_input("Impaciencia ρ", min_value=0.001, max_value=0.50, value=0.04, step=0.005, format="%.3f")
    theta = st.number_input("Inversa de la EIS θ", min_value=0.10, max_value=20.0, value=4.0, step=0.10)
    delta = st.number_input("Depreciación δ", min_value=0.0, max_value=0.99, value=0.05, step=0.01)
    n = st.number_input("Crecimiento poblacional n", min_value=0.0, max_value=0.49, value=0.01, step=0.005, format="%.3f")
    T = st.slider("Horizonte T", min_value=10, max_value=150, value=100, step=5)

params = RamseyParameters(A=A, alpha=alpha, rho=rho, theta=theta, delta=delta, n=n)
parameter_errors = validate_parameters(params, T=T)

if parameter_errors:
    for error in parameter_errors:
        st.error(error)
    st.stop()

try:
    steady_state = compute_steady_state(params)
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

try:
    c_0_local = compute_local_consumption(k_0, steady_state, linear_analysis)
    backward = compute_backward_stable_path(k_0, params, steady_state, linear_analysis)
    c_0_backward = backward["c_0_backward"]
    bvp = compute_bvp_path(k_0, params, steady_state, linear_analysis, T=T)
    c_0_bvp = bvp["c_0_bvp"]
except Exception as error:
    st.error(f"No fue posible calcular la senda estable para esta parametrización: {error}")
    st.stop()

with st.sidebar:
    consumption_mode = st.selectbox(
        "Valor inicial de consumo",
        [
            "Ingresar manualmente",
            "Usar aproximación backward",
            "Usar aproximación local",
            "Usar aproximación BVP"
        ]
    )
    suggested_default = float(max(c_0_backward, 0.0001))
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
metric_columns[4].metric("c₀ backward", f"{c_0_backward:.9f}")
metric_columns[5].metric("c₀ BVP", f"{c_0_bvp:.9f}")

if diagnosis["classification"] == "Descapitalización":
    st.error(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')
elif diagnosis["classification"] == "Sobreacumulación":
    st.warning(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')
elif diagnosis["classification"] == "Cercanía al equilibrio en el horizonte simulado":
    st.success(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')
else:
    st.info(f'{diagnosis["classification"]}: {diagnosis["explanation"]}')

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
        float(np.nanmax(result["k_path"])),
        float(np.nanmax(backward["k_path"]))
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
        phase_figure.add_trace(
            go.Scatter(
                x=backward["k_path"][::-1],
                y=backward["c_path"][::-1],
                mode="lines",
                name="Senda estable backward",
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
    zoom_c_min = max(0, min(c_0, steady_state.c_star, c_0_backward) - 0.05 * steady_state.c_star)
    zoom_c_max = max(c_0, steady_state.c_star, c_0_backward) + 0.05 * steady_state.c_star
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
    st.subheader("Comparación de aproximaciones de c₀")
    methods_table = pd.DataFrame({
        "Método": ["Lineal local", "Integración backward", "BVP"],
        "c₀ aproximado": [c_0_local, c_0_backward, c_0_bvp],
        "Diferencia respecto de backward": [
            c_0_local - c_0_backward,
            0.0,
            c_0_bvp - c_0_backward
        ]
    })
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
    st.caption("Backward y BVP aproximan la senda estable mediante procedimientos numéricos diferentes. Su cercanía funciona como control de robustez.")

with guide_tab:
    st.subheader("Ecuaciones del modelo")
    st.latex(r"\dot{k}=Ak^{\alpha}-c-(n+\delta)k")
    st.latex(r"\frac{\dot{c}}{c}=\frac{\alpha Ak^{\alpha-1}-\delta-\rho}{\theta}")
    st.subheader("Estado estacionario")
    st.latex(r"k^*=\left(\frac{\alpha A}{\rho+\delta}\right)^{\frac{1}{1-\alpha}}")
    st.latex(r"c^*=A(k^*)^{\alpha}-(n+\delta)k^*")
    st.subheader("Interpretación")
    st.write("El capital es una variable de estado predeterminada y el consumo es una variable de control que puede ajustarse inmediatamente.")
    st.write("La aproximación lineal describe el comportamiento local alrededor del estado estacionario. La integración backward y el BVP aproximan la senda estable no lineal.")
    st.write("La simulación forward reproduce la trayectoria generada por el consumo inicial elegido, aunque ese valor no coincida con la senda estable.")
    st.warning("En horizontes extensos, errores numéricos extremadamente pequeños pueden amplificarse debido al autovalor positivo propio de la estabilidad de punto silla.")
