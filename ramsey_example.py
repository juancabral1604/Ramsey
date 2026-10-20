from ramsey_core import RamseyParameters, build_scenario_bundle

params = RamseyParameters(
    A=1.0,
    alpha=0.33,
    rho=0.04,
    theta=2.0,
    delta=0.05,
    n=0.01
)

bundle = build_scenario_bundle(
    params=params,
    k_0_ratio=0.9,
    deviation=0.001,
    T=100
)

print("k* =", bundle["steady_state"].k_star)
print("c* =", bundle["steady_state"].c_star)
print("c0 local =", bundle["c_0_local"])
print("c0 backward =", bundle["c_0_backward"])
print("c0 BVP =", bundle["c_0_bvp"])
print(bundle["comparison"].to_string(index=False))
