from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, solve_ivp


@dataclass(frozen=True)
class RamseyParameters:
    A: float = 1.0
    alpha: float = 0.33
    rho: float = 0.04
    theta: float = 2.0
    delta: float = 0.05
    n: float = 0.01


@dataclass(frozen=True)
class RamseySteadyState:
    k_star: float
    c_star: float


def validate_parameters(params, k_0=None, c_0=None, T=None):
    errors = []
    if params.A <= 0:
        errors.append("A debe ser positivo.")
    if not 0 < params.alpha < 1:
        errors.append("alpha debe estar entre 0 y 1.")
    if params.rho <= 0:
        errors.append("rho debe ser positivo.")
    if params.theta <= 0:
        errors.append("theta debe ser positivo.")
    if not 0 <= params.delta < 1:
        errors.append("delta debe estar entre 0 y 1.")
    if params.n < 0:
        errors.append("n no puede ser negativo.")
    if params.rho <= params.n:
        errors.append("Para esta formulación se requiere rho mayor que n.")
    if k_0 is not None and k_0 <= 0:
        errors.append("k_0 debe ser positivo.")
    if c_0 is not None and c_0 <= 0:
        errors.append("c_0 debe ser positivo.")
    if T < 10 or T > 1000:
        errors.append("T debe estar entre 10 y 1000.")
    return errors


def production(k, params):
    return params.A * np.asarray(k) ** params.alpha


def marginal_product(k, params):
    return params.alpha * params.A * np.asarray(k) ** (params.alpha - 1)


def second_derivative_production(k, params):
    return params.alpha * params.A * (params.alpha - 1) * np.asarray(k) ** (params.alpha - 2)


def compute_steady_state(params):
    errors = validate_parameters(params)
    if errors:
        raise ValueError(" ".join(errors))
    k_star = (params.alpha * params.A / (params.rho + params.delta)) ** (1 / (1 - params.alpha))
    c_star = production(k_star, params) - (params.n + params.delta) * k_star
    if c_star <= 0:
        raise ValueError("La parametrización no genera un estado estacionario interior con consumo positivo.")
    return RamseySteadyState(float(k_star), float(c_star))


def capital_dot(k, c, params):
    return production(k, params) - c - (params.n + params.delta) * k


def consumption_dot(k, c, params):
    return (c / params.theta) * (marginal_product(k, params) - params.delta - params.rho)


def ramsey_system(t, values, params, k_floor=1e-8):
    k, c = values
    safe_k = max(float(k), k_floor)
    return np.array([
        capital_dot(safe_k, c, params),
        consumption_dot(safe_k, c, params)
    ])


def compute_linear_analysis(params, steady_state):
    k_star = steady_state.k_star
    c_star = steady_state.c_star
    J_star = np.array([
        [marginal_product(k_star, params) - (params.n + params.delta), -1.0],
        [(c_star / params.theta) * second_derivative_production(k_star, params),
         (marginal_product(k_star, params) - params.delta - params.rho) / params.theta]
    ], dtype=float)
    determinant = float(np.linalg.det(J_star))
    trace = float(np.trace(J_star))
    eigenvalues, eigenvectors = np.linalg.eig(J_star)
    eigenvalues = np.real_if_close(eigenvalues).astype(float)
    eigenvectors = np.real_if_close(eigenvectors).astype(float)
    stable_index = int(np.argmin(eigenvalues))
    unstable_index = int(np.argmax(eigenvalues))
    lambda_s = float(eigenvalues[stable_index])
    lambda_u = float(eigenvalues[unstable_index])
    v_s = eigenvectors[:, stable_index].astype(float)
    v_u = eigenvectors[:, unstable_index].astype(float)
    if v_s[0] < 0:
        v_s = -v_s
    if v_u[0] < 0:
        v_u = -v_u
    stable_slope = float(v_s[1] / v_s[0])
    unstable_slope = float(v_u[1] / v_u[0])
    theoretical_stable_slope = float(params.rho - params.n - lambda_s)
    return {
        "jacobian": J_star,
        "determinant": determinant,
        "trace": trace,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "lambda_s": lambda_s,
        "lambda_u": lambda_u,
        "v_s": v_s,
        "v_u": v_u,
        "stable_slope": stable_slope,
        "unstable_slope": unstable_slope,
        "theoretical_stable_slope": theoretical_stable_slope,
        "slopes_match": bool(np.isclose(stable_slope, theoretical_stable_slope))
    }


def compute_local_consumption(k_0, steady_state, linear_analysis):
    return float(
        steady_state.c_star
        + linear_analysis["stable_slope"] * (k_0 - steady_state.k_star)
    )


def compute_backward_stable_path(
    k_0,
    params,
    steady_state,
    linear_analysis,
    epsilon_ratio=1e-6,
    minimum_horizon=300,
    maximum_horizon=15000
):
    if k_0 <= 0:
        raise ValueError("k_0 debe ser positivo.")


    if np.isclose(k_0, steady_state.k_star):
        return {
            "success": True,
            "message": "k_0 coincide con k_star.",
            "k_0_backward": steady_state.k_star,
            "c_0_backward": steady_state.c_star,
            "time": np.array([0.0]),
            "k_path": np.array([steady_state.k_star]),
            "c_path": np.array([steady_state.c_star]),
            "adaptive_horizon": 0.0,
            "solution": None
        }


    lambda_s = linear_analysis["lambda_s"]
    v_s = linear_analysis["v_s"]


    if lambda_s >= 0:
        raise ValueError("No se identificó un autovalor estable negativo.")


    direction = np.sign(k_0 - steady_state.k_star)


    capital_scale = max(abs(steady_state.k_star), 1.0)


    displacement_size = (
        epsilon_ratio
        * capital_scale
        / max(abs(v_s[0]), 1e-14)
    )


    near_point = (
        np.array([
            steady_state.k_star,
            steady_state.c_star
        ])
        + direction * displacement_size * v_s
    )


    capital_distance = abs(
        k_0 - near_point[0]
    )


    initial_capital_displacement = abs(
        near_point[0] - steady_state.k_star
    )


    if capital_distance <= initial_capital_displacement:
        estimated_horizon = minimum_horizon
    else:
        estimated_horizon = (
            np.log(
                capital_distance
                / initial_capital_displacement
            )
            / abs(lambda_s)
        )


    adaptive_horizon = min(
        maximum_horizon,
        max(
            minimum_horizon,
            1.35 * estimated_horizon
        )
    )


    adaptive_max_step = max(
        0.1,
        adaptive_horizon / 5000
    )


    def reach_k0(t, values):
        return values[0] - k_0


    reach_k0.terminal = True
    reach_k0.direction = 0


    solution = solve_ivp(
        lambda t, values: ramsey_system(
            t,
            values,
            params
        ),
        [0, -adaptive_horizon],
        near_point,
        events=reach_k0,
        rtol=1e-10,
        atol=1e-12,
        max_step=adaptive_max_step
    )


    reached_k_0 = (
        len(solution.t_events[0]) > 0
    )


    if not reached_k_0:
        raise RuntimeError(
            "La integración backward no alcanzó k_0 "
            f"con un horizonte adaptativo de "
            f"{adaptive_horizon:.2f}."
        )


    event_values = solution.y_events[0][0]


    return {
        "success": bool(solution.success),
        "message": solution.message,
        "k_0_backward": float(
            event_values[0]
        ),
        "c_0_backward": float(
            event_values[1]
        ),
        "time": solution.t,
        "k_path": solution.y[0],
        "c_path": solution.y[1],
        "adaptive_horizon": float(
            adaptive_horizon
        ),
        "solution": solution
    }



def compute_bvp_path(
    k_0,
    params,
    steady_state,
    linear_analysis,
    T=100,
    nodes=500,
    tolerance=1e-8
):
    t_grid = np.linspace(0, T, nodes)
    lambda_s = linear_analysis["lambda_s"]
    stable_slope = linear_analysis["stable_slope"]
    decay = np.exp(lambda_s * t_grid)
    k_guess = steady_state.k_star + (k_0 - steady_state.k_star) * decay
    c_guess = steady_state.c_star + stable_slope * (k_0 - steady_state.k_star) * decay
    initial_guess = np.vstack((k_guess, c_guess))

    def system(t, values):
        k = np.maximum(values[0], 1e-10)
        c = values[1]
        k_dot = production(k, params) - c - (params.n + params.delta) * k
        c_dot = (c / params.theta) * (marginal_product(k, params) - params.delta - params.rho)
        return np.vstack((k_dot, c_dot))

    def boundary_conditions(initial_values, terminal_values):
        return np.array([
            initial_values[0] - k_0,
            terminal_values[1] - steady_state.c_star
            - stable_slope * (terminal_values[0] - steady_state.k_star)
        ])

    solution = solve_bvp(
        system,
        boundary_conditions,
        t_grid,
        initial_guess,
        tol=tolerance,
        max_nodes=20000,
        verbose=0
    )
    initial_values = solution.sol(0)
    terminal_values = solution.sol(T)
    boundary_residuals = boundary_conditions(initial_values, terminal_values)
    return {
        "success": bool(solution.success),
        "status": int(solution.status),
        "message": solution.message,
        "k_0_bvp": float(initial_values[0]),
        "c_0_bvp": float(initial_values[1]),
        "k_T_bvp": float(terminal_values[0]),
        "c_T_bvp": float(terminal_values[1]),
        "boundary_residuals": boundary_residuals,
        "time": t_grid,
        "k_path": solution.sol(t_grid)[0],
        "c_path": solution.sol(t_grid)[1],
        "solution": solution
    }


def simulate_ramsey(
    k_0,
    c_0,
    params,
    steady_state,
    T=100,
    points=1001,
    k_floor=1e-4
):
    errors = validate_parameters(params, k_0=k_0, c_0=c_0, T=T)
    if errors:
        raise ValueError(" ".join(errors))

    def capital_depletion(t, values):
        return values[0] - k_floor

    capital_depletion.terminal = True
    capital_depletion.direction = -1
    time_grid = np.linspace(0, T, points)
    solution = solve_ivp(
        lambda t, y: ramsey_system(t, y, params, k_floor=k_floor),
        [0, T],
        [k_0, c_0],
        t_eval=time_grid,
        events=capital_depletion,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1
    )
    time_path = solution.t
    k_path = solution.y[0]
    c_path = solution.y[1]
    capital_exhausted = len(solution.t_events[0]) > 0
    if capital_exhausted:
        event_time = float(solution.t_events[0][0])
        event_k = float(solution.y_events[0][0][0])
        event_c = float(solution.y_events[0][0][1])
        if not np.isclose(time_path[-1], event_time):
            time_path = np.append(time_path, event_time)
            k_path = np.append(k_path, event_k)
            c_path = np.append(c_path, event_c)
    else:
        event_time = np.nan
        event_k = np.nan
        event_c = np.nan
    distance_path = np.sqrt(
        (k_path - steady_state.k_star) ** 2
        + (c_path - steady_state.c_star) ** 2
    )
    closest_index = int(np.argmin(distance_path))
    return {
        "solution": solution,
        "time": time_path,
        "k_path": k_path,
        "c_path": c_path,
        "distance_path": distance_path,
        "integration_success": bool(solution.success),
        "solver_message": solution.message,
        "capital_exhausted": capital_exhausted,
        "event_time": event_time,
        "event_k": event_k,
        "event_c": event_c,
        "final_k": float(k_path[-1]),
        "final_c": float(c_path[-1]),
        "final_distance": float(distance_path[-1]),
        "minimum_distance": float(distance_path[closest_index]),
        "closest_time": float(time_path[closest_index]),
        "closest_k": float(k_path[closest_index]),
        "closest_c": float(c_path[closest_index])
    }


def classify_trajectory(result, steady_state):
    equilibrium_scale = np.hypot(steady_state.k_star, steady_state.c_star)
    relative_final_distance = result["final_distance"] / equilibrium_scale
    distance_path = result["distance_path"]
    comparison_index = max(0, len(distance_path) - max(2, len(distance_path) // 10))
    recent_distance = distance_path[comparison_index]
    approaching_at_end = result["final_distance"] < recent_distance
    if result["capital_exhausted"]:
        classification = "Descapitalización"
        explanation = "El capital alcanzó el límite mínimo de la simulación."
    elif relative_final_distance < 0.002 and approaching_at_end and result["integration_success"]:
        classification = "Cercanía al equilibrio en el horizonte simulado"
        explanation = "La trayectoria termina muy cerca del estado estacionario y continúa aproximándose."
    elif approaching_at_end and result["integration_success"]:
        classification = "Aproximación al equilibrio sin alcanzar la tolerancia"
        explanation = (
            "La trayectoria continúa aproximándose al estado estacionario, "
            "pero no alcanza la tolerancia relativa del 0,2 % dentro del "
            "horizonte simulado. Esto no demuestra convergencia asintótica."
        )
    elif (
        result["final_k"] > steady_state.k_star
        and result["final_c"] < steady_state.c_star
        and result["final_distance"] > result["minimum_distance"]
    ):
        classification = "Sobreacumulación"
        explanation = "El capital crece excesivamente y el consumo se aleja por debajo de su nivel estacionario."
    elif (
        result["final_k"] < steady_state.k_star
        and result["final_c"] > steady_state.c_star
        and result["final_distance"] > result["minimum_distance"]
    ):
        classification = "Descapitalización"
        explanation = "El capital disminuye y el consumo se aleja por encima de su nivel estacionario."
    else:
        classification = "Trayectoria no concluyente"
        explanation = "El horizonte simulado no permite clasificar la trayectoria con seguridad."
    return {
        "classification": classification,
        "explanation": explanation,
        "relative_final_distance": float(relative_final_distance),
        "approaching_at_end": bool(approaching_at_end)
    }


def shadow_price_path(c_path, theta):
    safe_consumption = np.maximum(np.asarray(c_path), 1e-12)
    return safe_consumption ** (-theta)


def build_scenario_bundle(
    params=RamseyParameters(),
    k_0_ratio=0.9,
    deviation=0.001,
    T=100
):
    steady_state = compute_steady_state(params)
    k_0 = k_0_ratio * steady_state.k_star
    linear_analysis = compute_linear_analysis(params, steady_state)
    c_0_local = compute_local_consumption(
        k_0, steady_state, linear_analysis
    )

    backward = None
    backward_error = None

    try:
        backward = compute_backward_stable_path(
            k_0, params, steady_state, linear_analysis
        )
    except Exception as error:
        backward_error = str(error)

    bvp = None
    bvp_error = None

    try:
        bvp = compute_bvp_path(
            k_0, params, steady_state, linear_analysis, T=T
        )
        if not bvp["success"]:
            bvp_error = bvp["message"]
            bvp = None
    except Exception as error:
        bvp_error = str(error)

    c_0_backward = (
        backward["c_0_backward"]
        if backward is not None else np.nan
    )
    c_0_bvp = (
        bvp["c_0_bvp"]
        if bvp is not None else np.nan
    )

    if backward is not None:
        c_0_reference = c_0_backward
        stable_method = "Backward adaptativo"
    elif bvp is not None:
        c_0_reference = c_0_bvp
        stable_method = "BVP de respaldo"
    else:
        c_0_reference = c_0_local
        stable_method = "Lineal local"

    c_0_low = c_0_reference * (1 - deviation)
    c_0_high = c_0_reference * (1 + deviation)

    low_result = simulate_ramsey(
        k_0, c_0_low, params, steady_state, T=T
    )
    correct_result = simulate_ramsey(
        k_0, c_0_reference, params, steady_state, T=T
    )
    high_result = simulate_ramsey(
        k_0, c_0_high, params, steady_state, T=T
    )

    low_diagnosis = classify_trajectory(low_result, steady_state)
    correct_diagnosis = classify_trajectory(correct_result, steady_state)
    high_diagnosis = classify_trajectory(high_result, steady_state)

    comparison = pd.DataFrame({
        "Escenario": [
            "Consumo bajo", "Consumo de referencia", "Consumo alto"
        ],
        "c0": [c_0_low, c_0_reference, c_0_high],
        "Capital final": [
            low_result["final_k"],
            correct_result["final_k"],
            high_result["final_k"]
        ],
        "Consumo final": [
            low_result["final_c"],
            correct_result["final_c"],
            high_result["final_c"]
        ],
        "Distancia mínima": [
            low_result["minimum_distance"],
            correct_result["minimum_distance"],
            high_result["minimum_distance"]
        ],
        "Distancia final": [
            low_result["final_distance"],
            correct_result["final_distance"],
            high_result["final_distance"]
        ],
        "Clasificación": [
            low_diagnosis["classification"],
            correct_diagnosis["classification"],
            high_diagnosis["classification"]
        ]
    })

    return {
        "params": params,
        "steady_state": steady_state,
        "linear_analysis": linear_analysis,
        "k_0": k_0,
        "c_0_local": c_0_local,
        "backward": backward,
        "backward_error": backward_error,
        "bvp": bvp,
        "bvp_error": bvp_error,
        "stable_method": stable_method,
        "c_0_reference": c_0_reference,
        "c_0_backward": c_0_backward,
        "c_0_bvp": c_0_bvp,
        "c_0_low": c_0_low,
        "c_0_high": c_0_high,
        "low_result": low_result,
        "correct_result": correct_result,
        "high_result": high_result,
        "low_diagnosis": low_diagnosis,
        "correct_diagnosis": correct_diagnosis,
        "high_diagnosis": high_diagnosis,
        "comparison": comparison
    }
