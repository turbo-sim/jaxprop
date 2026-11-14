"""Native complete-state workloads and standalone compiled consumers.

The matched PG cohort reproduces the published model's actual constants using
the current model's validated public calibration API. Its complete return is
still materialized: the 13 common fields constrain verification only. Separate
default-calibration cases retain each release's ordinary physical model.
"""

from uuid import uuid4

PG_FIELDS = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "heat_capacity_ratio",
    "speed_of_sound",
    "viscosity",
    "conductivity",
    "compressibility_factor",
    "gruneisen",
)
REAL_FIELDS = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "internal_energy",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "heat_capacity_ratio",
    "speed_of_sound",
    "viscosity",
    "conductivity",
)


def register(ctx):
    jp, CP, jax, jnp, np, api = ctx.jp, ctx.CP, ctx.jax, ctx.jnp, ctx.np, ctx.api
    pg = ctx.setup(
        "pg_matched_construction", lambda: api.make_pg("CO2", 600.0, 1e5, matched=True)
    )
    default_pg = ctx.setup(
        "pg_default_construction", lambda: api.make_pg("CO2", 600.0, 1e5)
    )
    expected = api.legacy_pg_constants("CO2", 600.0, 1e5)
    actual = {name: float(getattr(pg.constants, name)) for name in expected}
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-12, atol=1e-13)
    ctx.check("pg_matched_nine_constants", actual)

    def table(name, backend, label, *, water=False, size=16):
        # A unique empty directory makes the first observation genuinely fresh;
        # all subsequent load observations use its independently verified cache.
        directory = ctx.directory / (label + "_" + uuid4().hex)
        options = dict(
            h_min=400e3 if water else 600e3,
            h_max=550e3 if water else 850e3,
            p_min=2e6 if water else 5e6,
            p_max=4e6 if water else 12e6,
            N_h=size,
            N_p=size,
            N_p_sat=4,
            coarse_step=1,
            gradient_method="central",
            table_dir=str(directory),
        )
        factory = lambda: api.make_table(name, backend, options)
        model = ctx.setup(label + "_fresh_generation", factory, repeats=1)
        ctx.setup(label + "_cached_loading", factory, repeats=5)
        return model

    bicubic = table("CO2", "HEOS", "bicubic_CO2_HEOS")
    ctx.models["pg"], ctx.models["bicubic"] = pg, bicubic
    source = CP.AbstractState("HEOS", "CO2")
    fixtures = {}

    def coordinates(
        model,
        kind,
        shape,
        variant=0,
        *,
        scalar_pressure=False,
        water=False,
        backend="HEOS",
    ):
        key = id(model), shape, variant, scalar_pressure, water, backend
        if key in fixtures:
            return fixtures[key]
        fraction = np.linspace(0.0, 1.0, max(1, int(np.prod(shape)))).reshape(shape)
        pressure = (3e6 if water else 6e6) + 2e5 * (fraction + variant)
        if scalar_pressure:
            pressure = np.asarray(6e6 + 2e5 * variant)
        if kind == "pg":
            constants = model.constants
            R, gamma = float(constants.R), float(constants.gamma)
            cp = gamma * R / (gamma - 1)
            temperature = 600.0 + 10.0 * (fraction + variant)
            h_ref = (
                cp * float(constants.T_ref) if api.legacy else float(constants.h_ref)
            )
            enthalpy = h_ref + cp * (temperature - float(constants.T_ref))
            entropy = (
                float(constants.s_ref)
                + cp * np.log(temperature / float(constants.T_ref))
                - R * np.log(pressure / float(constants.P_ref))
            )
            density = pressure / (R * temperature)
        else:
            enthalpy = (450e3 if water else 680e3) + 10e3 * (fraction + variant)
            temperature, entropy, density = (np.empty(shape) for _ in range(3))
            raw = (
                source
                if backend == "HEOS" and not water
                else CP.AbstractState(backend, "Water" if water else "CO2")
            )
            p_values = np.broadcast_to(pressure, shape)
            for index in np.ndindex(shape):
                raw.update(
                    CP.HmassP_INPUTS, float(enthalpy[index]), float(p_values[index])
                )
                temperature[index], entropy[index], density[index] = (
                    raw.T(),
                    raw.smass(),
                    raw.rhomass(),
                )
        result = {
            name: jnp.asarray(value)
            for name, value in dict(
                pressure=pressure,
                temperature=temperature,
                enthalpy=enthalpy,
                entropy=entropy,
                density=density,
            ).items()
        }
        fixtures[key] = result
        return result

    def pair_inputs(values, pair):
        if pair == jp.PT_INPUTS:
            return values["pressure"], values["temperature"]
        if pair == jp.HmassP_INPUTS:
            return values["enthalpy"], values["pressure"]
        if pair == jp.PSmass_INPUTS:
            return values["pressure"], values["entropy"]
        raise ValueError("Unexpected primary native pair")

    def add_pair(
        label,
        model,
        kind,
        pair,
        shape,
        mode,
        *,
        scalar_pressure=False,
        fields=None,
        water=False,
        backend="HEOS",
        note="",
    ):
        evaluate = lambda first, second: api.evaluate(model, pair, first, second)
        function = jax.jit(evaluate) if mode == "compiled" else evaluate
        variants = [
            pair_inputs(
                coordinates(
                    model,
                    kind,
                    shape,
                    index,
                    scalar_pressure=scalar_pressure,
                    water=water,
                    backend=backend,
                ),
                pair,
            )
            for index in (0, 1)
        ]
        pair_name = {
            jp.PT_INPUTS: "PT",
            jp.HmassP_INPUTS: "HP",
            jp.PSmass_INPUTS: "PS",
        }[pair]
        shape_name = "scalar" if not shape else "batch" + str(int(np.prod(shape)))
        name = f"{label}_{mode}_{pair_name}_{shape_name}" + (
            "_scalar_pressure" if scalar_pressure else ""
        )
        known_exception = kind == "pg" and mode == "compiled" and shape == (1024,)
        ctx.add(
            name,
            function,
            variants,
            calls=12 if shape == (1024,) else 32,
            fields=fields or (PG_FIELDS if kind == "pg" else REAL_FIELDS),
            compiled=function if mode == "compiled" else None,
            budget=1.40 if known_exception else 1.10,
            note=note
            + (
                " Prior accepted Phase7 closed-model1024 full-state batch family, including PT/HP/PS and scalar/arraypressure; scope clarified before acceptance timing. Strict10% flag remains visible."
                if known_exception
                else ""
            ),
        )

    def validate_table(model, label, *, water=False, backend="HEOS"):
        """Check scalar fixtures against raw EOS and the public local derivative.

        Before timing, these 16/8-node studies declare source-relative rtol1e-3
        and AD-versus-centered-FD rtol1e-5. The latter checks the interpolant's
        actual derivative; source1/cp separately constrains its physical error.
        """
        raw = CP.AbstractState(backend, "Water" if water else "CO2")
        getters = {
            "pressure": raw.p,
            "temperature": raw.T,
            "density": raw.rhomass,
            "enthalpy": raw.hmass,
            "entropy": raw.smass,
            "internal_energy": raw.umass,
            "isobaric_heat_capacity": raw.cpmass,
            "isochoric_heat_capacity": raw.cvmass,
            "heat_capacity_ratio": lambda: raw.cpmass() / raw.cvmass(),
            "speed_of_sound": raw.speed_sound,
            "viscosity": raw.viscosity,
            "conductivity": raw.conductivity,
        }
        errors, derivative_checks = {name: [] for name in REAL_FIELDS}, []
        for index in (0, 1):
            values = coordinates(
                model, "bicubic", (), index, water=water, backend=backend
            )
            h, p = float(values["enthalpy"]), float(values["pressure"])
            raw.update(CP.HmassP_INPUTS, h, p)
            state = api.evaluate(
                model, jp.HmassP_INPUTS, values["enthalpy"], values["pressure"]
            )
            for name in REAL_FIELDS:
                actual_value, expected_value = (
                    float(api.raw(state, name)),
                    getters[name](),
                )
                assert np.isfinite(actual_value), (label, name)
                atol = (
                    1e-13
                    if name == "viscosity"
                    else 1e-10 if name == "conductivity" else 1e-6
                )
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=1e-3,
                    atol=atol,
                    err_msg=label + ": " + name,
                )
                errors[name].append(
                    abs(actual_value - expected_value) / max(abs(expected_value), 1e-30)
                )
            temperature = lambda target: api.raw(
                api.evaluate(model, jp.HmassP_INPUTS, target, values["pressure"]),
                "temperature",
            )
            derivative = float(
                jax.jvp(temperature, (values["enthalpy"],), (jnp.asarray(1.0),))[1]
            )
            step = 1e-5 * (model.h_max - model.h_min)
            difference = float(
                (
                    temperature(jnp.asarray(h + step))
                    - temperature(jnp.asarray(h - step))
                )
                / (2 * step)
            )
            assert np.isfinite(derivative) and derivative > 0
            np.testing.assert_allclose(
                derivative,
                difference,
                rtol=1e-5,
                atol=1e-12,
                err_msg=label + ": AD vs centered FD",
            )
            np.testing.assert_allclose(
                derivative,
                1.0 / raw.cpmass(),
                rtol=1e-3,
                atol=1e-12,
                err_msg=label + ": AD vs source1/cp",
            )
            derivative_checks.append(
                dict(AD=derivative, centered_FD=difference, source=1.0 / raw.cpmass())
            )
        return dict(
            points=2,
            source_rtol=1e-3,
            AD_FD_rtol=1e-5,
            max_property_relative_errors={
                name: max(values) for name, values in errors.items()
            },
            derivatives=derivative_checks,
        )

    ctx.validate(
        "bicubic_CO2_HEOS_independent_source_and_AD",
        lambda: validate_table(bicubic, "bicubic_CO2_HEOS"),
    )

    # Stratify shape coverage rather than duplicate the whole Cartesian product.
    for label, model, kind in (
        ("pg_matched", pg, "pg"),
        ("bicubic_CO2", bicubic, "bicubic"),
    ):
        for pair in (jp.PT_INPUTS, jp.HmassP_INPUTS, jp.PSmass_INPUTS):
            for mode in ("eager", "compiled"):
                add_pair(label, model, kind, pair, (), mode)
        for shape in ((1,), (16,), (1024,)):
            for mode in ("eager", "compiled"):
                if kind == "pg" and shape == (1,) and mode == "compiled":
                    continue
                add_pair(label, model, kind, jp.HmassP_INPUTS, shape, mode)
        for pair in (jp.PT_INPUTS, jp.PSmass_INPUTS):
            add_pair(label, model, kind, pair, (1024,), "compiled")
        if kind == "pg":
            add_pair(
                label,
                model,
                kind,
                jp.HmassP_INPUTS,
                (1024,),
                "compiled",
                scalar_pressure=True,
                note="Published pressure remains scalar; compare physical values by broadcasting only outside timing and retain actual returned shapes.",
            )

        # Independent axes expose actual broadcasting instead of pre-broadcast
        # input arrays. No returned legacy field is padded by this harness.
        for pair in (
            (jp.PT_INPUTS, jp.HmassP_INPUTS) if kind == "pg" else (jp.HmassP_INPUTS,)
        ):
            variants = []
            for index in (0, 1):
                pressure = jnp.linspace(6e6 + 1e5 * index, 6.4e6 + 1e5 * index, 16)[
                    None, :
                ]
                temperature = jnp.linspace(600.0 + index, 620.0 + index, 8)[:, None]
                if kind == "pg":
                    cp = pg.constants.gamma * pg.constants.R / (pg.constants.gamma - 1)
                    enthalpy = cp * temperature
                else:
                    enthalpy = jnp.linspace(
                        680e3 + 1e3 * index, 700e3 + 1e3 * index, 8
                    )[:, None]
                variants.append(
                    (pressure, temperature)
                    if pair == jp.PT_INPUTS
                    else (enthalpy, pressure)
                )
            evaluate = lambda first, second, model=model, pair=pair: api.evaluate(
                model, pair, first, second
            )
            for mode in ("eager", "compiled"):
                function = jax.jit(evaluate) if mode == "compiled" else evaluate
                ctx.add(
                    f"{label}_{mode}_{'PT' if pair == jp.PT_INPUTS else 'HP'}_broadcast8x16",
                    function,
                    variants,
                    fields=PG_FIELDS if kind == "pg" else REAL_FIELDS,
                    compiled=function if mode == "compiled" else None,
                )

        def complete_and_derivative(h, p, model=model):
            state = api.evaluate(model, jp.HmassP_INPUTS, h, p)
            derivative = jax.jvp(
                lambda value: api.raw(
                    api.evaluate(model, jp.HmassP_INPUTS, value, p), "temperature"
                ),
                (h,),
                (jnp.ones_like(h),),
            )[1]
            return {"state": state, "dTdh": derivative}

        function = jax.jit(complete_and_derivative)
        inputs = [
            pair_inputs(coordinates(model, kind, (), index), jp.HmassP_INPUTS)
            for index in (0, 1)
        ]
        ctx.add(
            label + "_compiled_fullstate_and_dTdh",
            function,
            inputs,
            fields=PG_FIELDS if kind == "pg" else REAL_FIELDS,
            compiled=function,
            category="full_state_derivative",
        )

        # Dynamic model buffers are an explicitly separate workload from the
        # closed-model public application calls above.
        if kind == "pg":
            changed = ctx.eqx.tree_at(
                lambda value: value.constants.mu_ref,
                model,
                model.constants.mu_ref * 1.001,
            )
        elif api.legacy:
            names = tuple(sorted(model.table["viscosity"]))
            entries = tuple(model.table["viscosity"][name] for name in names)
            changed = ctx.eqx.tree_at(
                lambda value: tuple(value.table["viscosity"][name] for name in names),
                model,
                tuple(value * 1.001 for value in entries),
            )
        else:
            # Viscosity has one authoritative runtime coefficient array; its
            # source-node inspection snapshot is not used by interpolation.
            changed = ctx.eqx.tree_at(
                lambda value: value._table["viscosity"]["coeffs"],
                model,
                model._table["viscosity"]["coeffs"] * 1.001,
            )
        numerical, static = ctx.eqx.partition(model, ctx.eqx.is_array)
        changed_numerical, _ = ctx.eqx.partition(changed, ctx.eqx.is_array)
        for shape in ((), (1024,)):
            traces = []

            def numerical_state(buffers, h, p, static=static, traces=traces):
                traces.append(1)
                return api.evaluate(
                    ctx.eqx.combine(buffers, static), jp.HmassP_INPUTS, h, p
                )

            dynamic = jax.jit(numerical_state)
            inputs = pair_inputs(coordinates(model, kind, shape, 0), jp.HmassP_INPUTS)
            variants = [
                (buffers, *inputs) for buffers in (numerical, changed_numerical)
            ]
            case_name = (
                label + "_dynamic_model_HP_" + ("scalar" if not shape else "batch1024")
            )
            ctx.add(
                case_name,
                dynamic,
                variants,
                calls=12 if shape else 32,
                fields=PG_FIELDS if kind == "pg" else REAL_FIELDS,
                compiled=dynamic,
                category="dynamic_model",
                note="Actual viscosity parameters change by1.001 at fixed state inputs. Public Equinox partition/combine keeps old metadata strings static and numerical coefficients dynamic. Separate from the historical closed-model40% exception.",
            )

            def verify_dynamic(
                dynamic=dynamic, variants=variants, traces=traces, case_name=case_name
            ):
                before, after = (
                    jax.block_until_ready(dynamic(*arguments)) for arguments in variants
                )
                np.testing.assert_allclose(
                    api.raw(after, "viscosity"),
                    api.raw(before, "viscosity") * 1.001,
                    rtol=2e-12,
                    atol=1e-15,
                )
                for name in (
                    "pressure",
                    "temperature",
                    "density",
                    "enthalpy",
                    "entropy",
                ):
                    np.testing.assert_array_equal(
                        api.raw(before, name), api.raw(after, name)
                    )
                assert len(traces) == 1, (case_name, len(traces))
                return dict(
                    outer_traces=len(traces),
                    parameter_multiplier=1.001,
                    thermodynamic_fields_unchanged=True,
                )

            ctx.validate(case_name + "_parameter_and_trace_check", verify_dynamic)

    # The common projection diagnoses output/guard costs; it never replaces the
    # complete 22-field current model in the primary state workloads above.
    def common_pg(h, p):
        state = api.evaluate(pg, jp.HmassP_INPUTS, h, p)
        return {name: api.raw(state, name) for name in PG_FIELDS}

    function = jax.jit(common_pg)
    ctx.add(
        "pg_matched_compiled_common13_projection_batch1024",
        function,
        [
            pair_inputs(coordinates(pg, "pg", (1024,), index), jp.HmassP_INPUTS)
            for index in (0, 1)
        ],
        calls=12,
        fields=PG_FIELDS,
        compiled=function,
        category="projection",
        note="Secondary output projection; primary calls return the actual complete state.",
    )
    for shape, mode in (((), "eager"), ((1024,), "compiled")):
        add_pair(
            "pg_default",
            default_pg,
            "pg",
            jp.HmassP_INPUTS,
            shape,
            mode,
            fields=("pressure", "temperature", "density"),
            note="As-shipped calibration in each version. Source-consistent current enthalpy anchor and ideal-gas cp/gamma correction intentionally change caloric fields; physicalT,p,rho fixtures match.",
        )

    def machine(model, h, p, ratio, efficiency, expansion=False):
        inlet = api.evaluate(model, jp.HmassP_INPUTS, h, p)
        ideal = api.evaluate(
            model, jp.PSmass_INPUTS, p * ratio, api.raw(inlet, "entropy")
        )
        increment = api.raw(ideal, "enthalpy") - h
        actual_h = h + (efficiency * increment if expansion else increment / efficiency)
        outlet = api.evaluate(model, jp.HmassP_INPUTS, actual_h, p * ratio)
        return dict(
            work=(h - actual_h) if expansion else (actual_h - h),
            outlet_temperature=api.raw(outlet, "temperature"),
            outlet_pressure=api.raw(outlet, "pressure"),
            status_inlet=api.status(inlet),
            status_ideal=api.status(ideal),
            status_outlet=api.status(outlet),
        )

    for label, model, kind in (
        ("pg_matched", pg, "pg"),
        ("bicubic_CO2", bicubic, "bicubic"),
    ):
        for expansion in (False, True):
            function = jax.jit(
                lambda h, p, ratio, efficiency, model=model, expansion=expansion: machine(
                    model, h, p, ratio, efficiency, expansion
                )
            )
            variants = [
                (
                    *pair_inputs(coordinates(model, kind, (), index), jp.HmassP_INPUTS),
                    jnp.asarray(0.95 if expansion else 1.1),
                    jnp.asarray(0.85),
                )
                for index in (0, 1)
            ]
            ctx.add(
                label + ("_compiled_expander" if expansion else "_compiled_compressor"),
                function,
                variants,
                compiled=function,
                category="consumer",
                note="Real current status arrays are returned; published absence is recorded, never replaced with invented status.",
            )

        def heat_exchanger(h_cold, h_hot, pressure, duty, model=model):
            fraction = jnp.linspace(0.0, 1.0, 128)
            cold_h, hot_h = h_cold + fraction * duty, h_hot - (1.0 - fraction) * duty
            cold = api.evaluate(model, jp.HmassP_INPUTS, cold_h, pressure)
            hot = api.evaluate(model, jp.HmassP_INPUTS, hot_h, pressure * 1.05)
            temperature_difference = api.raw(hot, "temperature") - api.raw(
                cold, "temperature"
            )
            return dict(
                temperature_difference=temperature_difference,
                minimum_delta_T=jnp.min(temperature_difference),
                heat_absorbed=api.raw(cold, "enthalpy")[-1]
                - api.raw(cold, "enthalpy")[0],
                heat_released=api.raw(hot, "enthalpy")[-1]
                - api.raw(hot, "enthalpy")[0],
                status_cold=api.status(cold),
                status_hot=api.status(hot),
            )

        function = jax.jit(heat_exchanger)
        if kind == "pg":
            cp = float(pg.constants.gamma * pg.constants.R / (pg.constants.gamma - 1))
            enthalpies = (cp * 600.0, cp * 660.0)
        else:
            enthalpies = (670e3, 730e3)
        variants = [
            (
                jnp.asarray(enthalpies[0] + index * 1000.0),
                jnp.asarray(enthalpies[1] + index * 1000.0),
                jnp.asarray(8e6),
                jnp.asarray(20e3),
            )
            for index in (0, 1)
        ]
        ctx.add(
            label + "_compiled_heat_exchanger_128cells",
            function,
            variants,
            compiled=function,
            category="consumer",
            note="Two full public batch calls; numerical output and both real current status arrays are retained.",
        )

    water = table("Water", "HEOS", "bicubic_Water_HEOS", water=True, size=8)
    add_pair(
        "bicubic_Water", water, "bicubic", jp.HmassP_INPUTS, (), "eager", water=True
    )
    add_pair(
        "bicubic_Water", water, "bicubic", jp.PT_INPUTS, (), "compiled", water=True
    )
    ctx.validate(
        "bicubic_Water_HEOS_independent_source_and_AD",
        lambda: validate_table(water, "bicubic_Water_HEOS", water=True),
    )
    try:
        CP.AbstractState("REFPROP", "CO2")
    except (ValueError, RuntimeError) as error:
        ctx.skip("bicubic_CO2_REFPROP", "Optional source unavailable: " + str(error))
    else:
        refprop = table("CO2", "REFPROP", "bicubic_CO2_REFPROP", size=8)
        add_pair(
            "bicubic_CO2_REFPROP",
            refprop,
            "bicubic",
            jp.HmassP_INPUTS,
            (),
            "eager",
            backend="REFPROP",
        )
        add_pair(
            "bicubic_CO2_REFPROP",
            refprop,
            "bicubic",
            jp.PSmass_INPUTS,
            (),
            "compiled",
            backend="REFPROP",
        )

        ctx.validate(
            "bicubic_CO2_REFPROP_independent_source_and_AD",
            lambda: validate_table(refprop, "bicubic_CO2_REFPROP", backend="REFPROP"),
        )

    if api.legacy:
        for pair in ("PT", "HP"):
            ctx.skip(
                "binary_native_pgpg_compiled_" + pair,
                "Published package has no corresponding native fixed-composition binary evaluator.",
            )
    else:
        other = api.make_pg("Nitrogen", 600.0, 1e5, matched=True)
        binary = ctx.setup(
            "binary_native_pgpg_construction",
            lambda: api.make_binary((pg, other), (0.3, 0.7)),
        )
        for pair in (jp.PT_INPUTS, jp.HmassP_INPUTS):
            if pair == jp.PT_INPUTS:
                variants = [
                    (jnp.asarray(1e6), jnp.asarray(600.0 + index * 10))
                    for index in (0, 1)
                ]
            else:
                cp = sum(
                    weight
                    * float(
                        component.constants.gamma
                        * component.constants.R
                        / (component.constants.gamma - 1)
                    )
                    for component, weight in zip((pg, other), (0.3, 0.7))
                )
                variants = [
                    (jnp.asarray(cp * (600.0 + index * 10)), jnp.asarray(1e6))
                    for index in (0, 1)
                ]
            function = jax.jit(
                lambda first, second, pair=pair: api.evaluate(
                    binary, pair, first, second
                )
            )
            ctx.add(
                "binary_native_pgpg_compiled_"
                + ("PT" if pair == jp.PT_INPUTS else "HP"),
                function,
                variants,
                compiled=function,
                category="new_only",
                comparable=False,
                note="Absolute cost only; no published native-binary counterpart and no speedup ratio.",
            )

    def verify_matched_pg():
        state = api.evaluate(pg, jp.PT_INPUTS, jnp.asarray(8e6), jnp.asarray(610.0))
        cp = actual["gamma"] * actual["R"] / (actual["gamma"] - 1)
        np.testing.assert_allclose(api.raw(state, "enthalpy"), cp * 610.0, rtol=2e-13)
        np.testing.assert_allclose(
            api.raw(state, "density"), 8e6 / (actual["R"] * 610.0), rtol=2e-13
        )
        derivative = jax.jvp(
            lambda h: api.raw(
                api.evaluate(pg, jp.HmassP_INPUTS, h, jnp.asarray(8e6)), "temperature"
            ),
            (jnp.asarray(cp * 610.0),),
            (jnp.asarray(1.0),),
        )[1]
        np.testing.assert_allclose(derivative, 1.0 / cp, rtol=2e-13)

    ctx.validate("matched_pg_independent_equations", verify_matched_pg)
