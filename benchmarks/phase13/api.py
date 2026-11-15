"""Public-API selection for the unmodified published and checkout packages.

This adapter belongs to the benchmark, not to JaxProp or a consuming package.
It translates construction signatures and selects the appropriate public call
once per workload. It never changes the wheel, manufactures missing properties,
or pads the state returned by either implementation.
"""


class API:
    def __init__(self, jp, legacy, CP, jax, jnp, np, eqx):
        self.jp = jp
        self.legacy = bool(legacy)
        self.CP, self.jax, self.jnp, self.np, self.eqx = CP, jax, jnp, np, eqx
        self._legacy_calibrations = {}

    def make_source(self, name, backend="HEOS"):
        if self.legacy:
            return self.jp.Fluid(name=name, backend=backend)
        return self.jp.Fluid(name, method="coolprop", backend=backend)

    def make_callback(self, name, backend="HEOS"):
        if self.legacy:
            return self.jp.FluidJAX(name=name, backend=backend)
        return self.jp.Fluid(name, method="coolprop_jax", backend=backend)

    def legacy_pg_constants(self, name, T_ref, p_ref):
        """Independently reproduce the published model's calibration law.

        The published model combines the source cp/cv ratio with the universal
        gas constant and uses h=cp*T. The matched cohort adopts those *actual*
        constants in the current model, through its validated public update API.
        The separate default cohort keeps each release's ordinary calibration.
        """
        key = name, float(T_ref), float(p_ref)
        if key not in self._legacy_calibrations:
            source = self.CP.AbstractState("HEOS", name)
            source.update(self.CP.PT_INPUTS, float(p_ref), float(T_ref))
            R = self.jp.GAS_CONSTANT / source.molar_mass()
            gamma = source.cpmass() / source.cvmass()
            mu_ref, k_ref, s_ref = (
                source.viscosity(),
                source.conductivity(),
                source.smass(),
            )
            source.update(self.CP.PT_INPUTS, float(p_ref), float(T_ref + 100.0))

            def sutherland(reference, other):
                ratio = (other / reference) * ((T_ref + 100.0) / T_ref) ** (-1.5)
                return (ratio * (T_ref + 100.0) - T_ref) / (1.0 - ratio)

            self._legacy_calibrations[key] = dict(
                R=R,
                gamma=gamma,
                T_ref=float(T_ref),
                P_ref=float(p_ref),
                s_ref=s_ref,
                mu_ref=mu_ref,
                k_ref=k_ref,
                S_mu=sutherland(mu_ref, source.viscosity()),
                S_k=sutherland(k_ref, source.conductivity()),
            )
        return dict(self._legacy_calibrations[key])

    def make_pg(self, name, T_ref=600.0, p_ref=1e5, matched=False):
        if self.legacy:
            return self.jp.FluidPerfectGas(name, T_ref=T_ref, p_ref=p_ref)
        model = self.jp.Fluid(
            name,
            method="perfect_gas",
            model_options=dict(T_ref=T_ref, p_ref=p_ref),
        )
        if matched:
            constants = self.legacy_pg_constants(name, T_ref, p_ref)
            cp = constants["gamma"] * constants["R"] / (constants["gamma"] - 1)
            model = model.with_constants(**constants, h_ref=cp * T_ref)
        return model

    def make_table(self, name, backend, options):
        if self.legacy:
            return self.jp.FluidBicubic(name, backend, **dict(options))
        return self.jp.Fluid(
            name, method="bicubic", backend=backend, table_options=dict(options)
        )

    def evaluate(self, model, pair, first, second):
        if self.legacy or getattr(model, "provider", None) == "coolprop":
            return model.get_state(pair, first, second)
        return model.get_state(pair, first, second, errors="mask")

    def raw(self, state, canonical):
        if isinstance(state, dict):
            return state[canonical]
        if self.legacy:
            return getattr(state, canonical)
        return state.raw_property(canonical)

    def status(self, state):
        # Absence is a capability difference, not an invented success code.
        return getattr(state, "status", None)

    def dictionary(self, state):
        return dict(state) if isinstance(state, dict) else state.to_dict()

    def make_binary(self, components, mass_fractions):
        components, mass_fractions = tuple(components), tuple(mass_fractions)
        names = tuple(component.name for component in components)
        name = "_".join(names) + "_mixture"
        if not self.legacy:
            return self.jp.Fluid(
                name,
                method="binary_mixture",
                model_options=dict(
                    components=components, mass_fractions=mass_fractions
                ),
            )
        if not all(isinstance(component, self.jp.Fluid) for component in components):
            raise NotImplementedError(
                "The published matched binary route requires direct source components. "
                "Its experimental FluidTwoComponent depends on an absent MixtureState."
            )
        if components[0].backend != components[1].backend or mass_fractions[1] <= 0:
            raise NotImplementedError(
                "The published FluidMix comparison requires one source backend and a finite mass ratio."
            )
        return self.jp.FluidMix(
            name,
            backend=components[0].backend,
            mixture_ratio=mass_fractions[0] / mass_fractions[1],
        )


VersionAPI = API
