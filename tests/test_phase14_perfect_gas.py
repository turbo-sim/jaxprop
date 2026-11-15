"""Complete HP states and model derivatives across the native batch boundary.

The oracle uses independent ideal-gas equations and the conventional Sutherland
formula. It does not call the implementation's property or policy helpers.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp


@pytest.fixture(scope="module")
def gas():
    return jp.Fluid(
        "CO2", method="perfect_gas", model_options={"T_ref": 600.0, "p_ref": 1e5}
    )


def constants(model):
    return {name: float(value) for name, value in vars(model.constants).items()}


def independent_hp_values(c, enthalpy, pressure):
    """All 22 model fields at moderate, independently admissible HP inputs."""
    R, gamma = c["R"], c["gamma"]
    cp, cv = gamma * R / (gamma - 1), R / (gamma - 1)
    h, p = np.broadcast_arrays(enthalpy, pressure)
    T = c["T_ref"] + (h - c["h_ref"]) / cp
    one, zero = np.ones_like(T), np.zeros_like(T)

    def transport(reference, sutherland):
        return (
            reference
            * (T / c["T_ref"]) ** 1.5
            * (c["T_ref"] + sutherland)
            / (T + sutherland)
        )

    return {
        "temperature": T,
        "pressure": p,
        "density": p / (R * T),
        "enthalpy": h,
        "internal_energy": h - R * T,
        "entropy": c["s_ref"]
        + cp * np.log(T / c["T_ref"])
        - R * np.log(p / c["P_ref"]),
        "viscosity": transport(c["mu_ref"], c["S_mu"]),
        "conductivity": transport(c["k_ref"], c["S_k"]),
        "speed_of_sound": np.sqrt(gamma * R * T),
        "heat_capacity_ratio": gamma * one,
        "isobaric_heat_capacity": cp * one,
        "isochoric_heat_capacity": cv * one,
        "compressibility_factor": one,
        "gruneisen": (gamma - 1) * one,
        "is_two_phase": zero,
        "isobaric_expansion_coefficient": 1 / T,
        "isothermal_compressibility": 1 / p,
        "isentropic_compressibility": 1 / (gamma * p),
        "isothermal_bulk_modulus": p,
        "isentropic_bulk_modulus": gamma * p,
        "joule_thomson": zero,
        "isothermal_joule_thomson": zero,
    }


def batch_inputs(model, size):
    c = constants(model)
    cp = c["gamma"] * c["R"] / (c["gamma"] - 1)
    T = np.linspace(450.0, 750.0, size)
    h = c["h_ref"] + cp * (T - c["T_ref"])
    p = np.linspace(7e6, 9e6, size)
    return c, h, p


@pytest.mark.parametrize("size", [63, 64, 65, 128, 1024])
@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_complete_hp_batches_keep_equations_status_and_failed_input_gradients(
    gas, size, errors
):
    c, h, p = batch_inputs(gas, size)
    supplied_h, supplied_p = h.copy(), p.copy()
    supplied_h[1], supplied_p[-1] = np.nan, -1.0
    oracle_p = p.copy()
    if errors == "clip":
        # At these temperatures the declared pressure floor is the active
        # projection; the density floor would require a smaller pressure.
        oracle_p[-1] = 0.1
    expected = independent_hp_values(c, h, oracle_p)
    failed = np.zeros(size, dtype=bool)
    failed[1] = True
    if errors == "mask":
        failed[-1] = True

    evaluate = jax.jit(
        lambda h, p: gas.get_state(jp.HmassP_INPUTS, h, p, errors=errors)
    )
    state = evaluate(jnp.asarray(supplied_h), jnp.asarray(supplied_p))
    assert state.available_properties == frozenset(expected)
    assert state.shape == (size,) and state.status.dtype == jnp.int32
    np.testing.assert_array_equal(state.status, failed.astype(np.int32))
    clipped = np.zeros(size, dtype=bool)
    clipped[-1] = errors == "clip"
    np.testing.assert_array_equal(state.clipped, clipped)
    for name, value in expected.items():
        oracle = np.where(failed, np.nan, value)
        np.testing.assert_allclose(
            state.raw_property(name), oracle, rtol=3e-12, atol=1e-12, err_msg=name
        )

    derivative = jax.jit(
        jax.grad(
            lambda h: jnp.nan_to_num(
                evaluate(h, jnp.asarray(supplied_p)).raw_property("temperature")
            ).sum()
        )
    )(jnp.asarray(supplied_h))
    cp = c["gamma"] * c["R"] / (c["gamma"] - 1)
    np.testing.assert_allclose(
        derivative, np.where(failed, 0.0, 1 / cp), rtol=3e-12, atol=1e-14
    )


@pytest.mark.parametrize("parameter", ["gamma", "h_ref"])
@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_complete_hp_model_jvp_and_vjp_use_current_dynamic_parameters(
    gas, parameter, errors
):
    c, h, p = batch_inputs(gas, 128)
    supplied_h = h.copy()
    supplied_h[1] = np.nan
    expected = independent_hp_values(c, h, p)
    names = tuple(expected)
    scales = np.asarray([max(float(np.max(np.abs(expected[n]))), 1e-10) for n in names])
    weights = np.linspace(0.5, 1.5, len(names))

    def evaluate(value):
        current = eqx.tree_at(
            lambda model: getattr(model.constants, parameter), gas, value
        )
        state = current.get_state(
            jp.HmassP_INPUTS, jnp.asarray(supplied_h), jnp.asarray(p), errors=errors
        )
        complete = jnp.stack(tuple(state.raw_property(n) for n in names), axis=-1)
        return jnp.nan_to_num(complete / jnp.asarray(scales))

    value = getattr(gas.constants, parameter)
    _, forward = jax.jit(
        lambda value: jax.jvp(evaluate, (value,), (jnp.ones_like(value),))
    )(value)
    reverse = jax.jit(
        jax.grad(lambda value: (evaluate(value) * jnp.asarray(weights)).sum())
    )(value)

    # Differentiate all property definitions by an independent centered finite
    # difference at fixed HP, rather than reusing policy/interpolation helpers.
    step = max(abs(c[parameter]), 1.0) * 1e-5
    lower = independent_hp_values({**c, parameter: c[parameter] - step}, h, p)
    upper = independent_hp_values({**c, parameter: c[parameter] + step}, h, p)
    expected_derivative = (
        np.stack([(upper[n] - lower[n]) / (2 * step) for n in names], axis=-1) / scales
    )
    expected_derivative[1] = 0.0
    assert np.all(np.isfinite(forward)) and np.isfinite(reverse)
    np.testing.assert_allclose(forward, expected_derivative, rtol=3e-7, atol=3e-10)
    np.testing.assert_allclose(
        reverse, np.sum(expected_derivative * weights), rtol=3e-7, atol=3e-10
    )


@pytest.mark.parametrize("size", [63, 64, 128, 1024])
@pytest.mark.parametrize("cohort", ["valid", "all_failed"])
@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_complete_hp_uniform_cohorts_keep_status_and_zero_failed_gradients(
    gas, size, cohort, errors
):
    c, h, p = batch_inputs(gas, size)
    expected = independent_hp_values(c, h, p)
    supplied_h = np.full_like(h, np.nan) if cohort == "all_failed" else h
    failed = cohort == "all_failed"

    def evaluate(enthalpy):
        return gas.get_state(jp.HmassP_INPUTS, enthalpy, jnp.asarray(p), errors=errors)

    state = jax.jit(evaluate)(jnp.asarray(supplied_h))
    assert state.available_properties == frozenset(expected)
    np.testing.assert_array_equal(
        state.status, np.full(size, int(failed), dtype=np.int32)
    )
    np.testing.assert_array_equal(state.clipped, np.zeros(size, dtype=bool))
    for name, value in expected.items():
        np.testing.assert_allclose(
            state.raw_property(name),
            np.full(size, np.nan) if failed else value,
            rtol=3e-12,
            atol=1e-12,
            err_msg=name,
        )

    derivative = jax.jit(
        jax.grad(
            lambda enthalpy: jnp.nan_to_num(
                evaluate(enthalpy).raw_property("temperature")
            ).sum()
        )
    )(jnp.asarray(supplied_h))
    cp = c["gamma"] * c["R"] / (c["gamma"] - 1)
    np.testing.assert_allclose(
        derivative,
        np.full(size, 0.0 if failed else 1 / cp),
        rtol=3e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_compiled_multi_pair_states_reuse_dynamic_model_calibration(gas, errors):
    """One compiled consumer may use several pairs and updated model values."""
    traces = []

    @eqx.filter_jit
    def recover(model, temperature, pressure):
        traces.append(True)
        direct = model.get_state(jp.PT_INPUTS, pressure, temperature, errors=errors)
        from_h = model.get_state(
            jp.HmassP_INPUTS,
            direct.raw_property("enthalpy"),
            direct.raw_property("pressure"),
            errors=errors,
        )
        from_s = model.get_state(
            jp.PSmass_INPUTS,
            direct.raw_property("pressure"),
            direct.raw_property("entropy"),
            errors=errors,
        )
        return direct, from_h, from_s

    original_temperature = np.linspace(450.0, 750.0, 128)
    original_pressure = np.linspace(7e6, 9e6, 128)
    supplied_temperature = original_temperature.copy()
    supplied_pressure = original_pressure.copy()
    supplied_temperature[1], supplied_pressure[-1] = np.nan, -1.0
    changed = eqx.tree_at(
        lambda model: (model.constants.gamma, model.constants.h_ref),
        gas,
        (gas.constants.gamma * 1.01, gas.constants.h_ref + 200.0),
    )
    for model in (gas, changed):
        c = constants(model)
        cp = c["gamma"] * c["R"] / (c["gamma"] - 1)
        h = c["h_ref"] + cp * (original_temperature - c["T_ref"])
        expected_pressure = original_pressure.copy()
        failed = np.zeros(128, dtype=bool)
        failed[1] = True
        if errors == "mask":
            failed[-1] = True
        else:
            expected_pressure[-1] = 0.1
        expected = independent_hp_values(c, h, expected_pressure)
        states = recover(
            model, jnp.asarray(supplied_temperature), jnp.asarray(supplied_pressure)
        )
        for index, state in enumerate(states):
            np.testing.assert_array_equal(state.status, failed.astype(np.int32))
            clipped = np.zeros(128, dtype=bool)
            if errors == "clip" and index == 0:
                clipped[-1] = True
            np.testing.assert_array_equal(state.clipped, clipped)
            for name, values in expected.items():
                np.testing.assert_allclose(
                    state.raw_property(name),
                    np.where(failed, np.nan, values),
                    rtol=3e-12,
                    atol=1e-12,
                    err_msg=name,
                )
    assert len(traces) == 1
