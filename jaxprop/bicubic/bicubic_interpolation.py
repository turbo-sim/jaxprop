import os
import time
import pickle
import copy
import hashlib
import inspect
from pathlib import Path
from numbers import Real
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import CoolProp.CoolProp as CP
import jaxprop.coolprop as jxp
from tqdm import tqdm

from ..errors import CalculationError, InvalidConfigurationError, UnsupportedOptionError
from ..configuration import (
    build_configuration,
    FluidCapabilities,
    FluidDomain,
    FrozenMapping,
)
from ..fluid_api import SOURCE_PROPERTIES, LEGACY_MIXTURE_PROPERTIES
from ..source_references import (
    SourceReferencesMixin,
    initialize_source_references,
    composite_source_references,
)
from .cache_metadata import metadata_fingerprint, source_signature
from .phase_envelope import PHASE_ENVELOPE_SETTINGS, source_phase_cell_validity
from .runtime import (
    InspectionPayload,
    inspection_view,
    prepare_runtime,
    prepare_saturation,
)
from .numerics import interpolate_properties
from .request_policy import (
    checked_bicubic_jit,
    checked_saturation_jit,
    evaluate_scalar,
    evaluate_state,
    evaluate_saturation,
)
from .table_cache import (
    CACHE_FORMAT_VERSION,
    GENERATOR_REVISION,
    normalize_generated_arrays,
    save_table_atomic,
    table_validation_error,
)


class FluidBicubic(SourceReferencesMixin, eqx.Module):
    r"""
    Fluid model using bicubic property interpolation on an enthalpy-pressure grid.

    Workflow:
      * On construction, attempts to load a precomputed table from disk.
      * If not found, generates the table with finite-difference derivatives,
        saves it to disk, and stores it in memory.
      * On `get_state`, performs bicubic interpolation at the requested state.

    Generation uses the selected finite-difference scheme. Cache reuse requires
    an exact generation manifest, verified source identity and valid arrays.

    Parameters
    ----------
    fluid_name : str
        Fluid identifier for CoolProp.
    backend : str
        CoolProp backend string.
    h_min, h_max : float
        Min/max enthalpy [J/kg].
    p_min, p_max : float
        Min/max pressure [Pa].
    N_h, N_p : int
        Number of grid points in h and p.
    coarse_step : int
        Number of points to skip when determining the initial guess during 2D inverse interpolations
    inverse_max_steps : int, optional
        Nonnegative iteration limit for property-pair inverses (default: 64).
    inverse_tolerance : float, optional
        Positive tolerance for the nonlinear solver and its independent scaled
        residual check (default: 1e-10). These inverse controls do not change
        table generation or the persisted cache identity.
    table_name : str, optional
        File stem. The default includes a digest of the complete generation
        identity so different configurations can coexist in one cache directory.
    table_dir : str, optional
        Directory for saving/loading table pickle (default: "fluid_tables").
    identifier : str, optional
        Tag stored in the returned FluidState objects.

    """

    # Attributes (static vs dynamic)
    fluid_name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    h_min: float = eqx.field(static=True)
    h_max: float = eqx.field(static=True)
    p_min: float = eqx.field(static=True)
    p_max: float = eqx.field(static=True)
    N_h: int = eqx.field(static=True)
    N_p: int = eqx.field(static=True)
    N_p_sat: int = eqx.field(static=True)
    coarse_step: int = eqx.field(static=True)
    inverse_max_steps: int = eqx.field(static=True)
    inverse_tolerance: float = eqx.field(static=True)
    delta_h: float = eqx.field(static=True)
    delta_logP: float = eqx.field(static=True)
    delta_logP_sat: float = eqx.field(static=True)
    h_vals: jnp.ndarray = eqx.field(static=False)
    logP_vals: jnp.ndarray = eqx.field(static=False)
    logP_sat_vals: jnp.ndarray = eqx.field(static=False)
    table_name: str = eqx.field(static=True)
    table_dir: str = eqx.field(static=True)
    _table: dict = eqx.field(static=False)
    _coefficient_validity: dict = eqx.field(static=False)
    _cell_validity: jax.Array = eqx.field(static=False)
    _inspection: object = eqx.field(static=True)
    _input_fallbacks: object = eqx.field(static=True)
    _saturation_table: dict = eqx.field(static=False)
    _saturation_validity: dict = eqx.field(static=False)
    _saturation_capabilities: object = eqx.field(static=True)
    _saturation_p_max: float = eqx.field(static=True)
    identifier: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    model: str = eqx.field(static=True)
    reference_id: tuple = eqx.field(static=True)
    grad_method: str = eqx.field(static=True)
    metadata: dict = eqx.field(static=True)
    table_mode: str = eqx.field(static=True)
    mixture_ratio: float = eqx.field(static=True)
    metastable_phase: str = eqx.field(static=True)
    _references: object = eqx.field(static=True)
    _configuration: object = eqx.field(static=True, default=None)
    _capabilities: object = eqx.field(static=True, default=None)

    def __init__(
        self,
        fluid_name: str = None,
        backend: str = "HEOS",
        h_min: float = None,
        h_max: float = None,
        p_min: float = None,
        p_max: float = None,
        N_h: int = None,
        N_p: int = None,
        N_p_sat: int = 60,
        mixture_ratio: float = None,
        metastable_phase: str = None,
        coarse_step: int = 10,
        gradient_method: str = "central",
        identifier: str = None,
        table_name: str = None,
        table_dir: str = "fluid_tables",
        *,
        name: str = None,
        inverse_max_steps: int = 64,
        inverse_tolerance: float = 1e-10,
    ):
        if name is not None:
            if fluid_name is not None and fluid_name != name:
                raise InvalidConfigurationError(
                    fluid_name=name,
                    provider="bicubic",
                    model="real_fluid",
                    request="name",
                    reason="name and fluid_name describe different sources.",
                )
            fluid_name = name
        backend = "HEOS" if backend is None else backend
        # Validate static configuration before allocating grids or evaluating
        # reference states. Invalid options must never produce an empty cache.
        context = dict(fluid_name=fluid_name, provider="bicubic", model="real_fluid")

        def invalid(request, reason):
            raise InvalidConfigurationError(request=request, reason=reason, **context)

        if not isinstance(fluid_name, str) or not fluid_name:
            invalid("fluid_name", "Provide a nonempty CoolProp source name.")
        if not isinstance(backend, str) or not backend:
            invalid("backend", "Provide a nonempty CoolProp source backend.")
        for name, value in (
            ("N_h", N_h),
            ("N_p", N_p),
            ("N_p_sat", N_p_sat),
            ("coarse_step", coarse_step),
        ):
            minimum = 1 if name == "coarse_step" else 2
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                invalid(name, f"Expected an integer >= {minimum}.")
        try:
            for value in (h_min, h_max, p_min, p_max):
                scalar = np.asarray(value)
                if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
                    invalid(
                        "bounds",
                        "Table bounds must be real finite scalars, not booleans or strings.",
                    )
            bounds = np.asarray([h_min, h_max, p_min, p_max], dtype=float)
        except (TypeError, ValueError) as error:
            raise InvalidConfigurationError(
                request="bounds",
                reason="Table bounds must be real finite scalars.",
                **context,
            ) from error
        h_min, h_max, p_min, p_max = map(float, bounds)
        if (
            not np.all(np.isfinite(bounds))
            or not h_min < h_max
            or not 0 < p_min < p_max
        ):
            invalid("bounds", "Require finite h_min < h_max and 0 < p_min < p_max.")
        if gradient_method not in ("forward", "central"):
            invalid("gradient_method", "Choose 'forward' or 'central'.")
        if (
            isinstance(inverse_max_steps, (bool, np.bool_))
            or not isinstance(inverse_max_steps, (int, np.integer))
            or inverse_max_steps < 0
        ):
            invalid(
                "inverse_max_steps",
                "Use a nonnegative integer inverse iteration budget.",
            )
        if (
            isinstance(inverse_tolerance, (bool, np.bool_))
            or not isinstance(inverse_tolerance, Real)
            or not np.isfinite(float(inverse_tolerance))
            or inverse_tolerance <= 0
        ):
            invalid(
                "inverse_tolerance", "Use a finite positive inverse residual tolerance."
            )
        if metastable_phase not in (None, "liquid", "vapor"):
            invalid("metastable_phase", "Choose None, 'liquid', or 'vapor'.")
        is_mixture = fluid_name.lower().endswith("_mixture")
        context["model"] = "binary_mixture" if is_mixture else "real_fluid"
        if is_mixture != (mixture_ratio is not None):
            invalid(
                "mixture_ratio",
                "A legacy binary '_mixture' source requires its mass ratio; ordinary sources do not accept it.",
            )
        if is_mixture:
            from ..binary_mixture.compat import fractions_from_ratio

            try:
                if np.asarray(mixture_ratio).ndim != 0:
                    invalid("mixture_ratio", "Provide a scalar mass ratio.")
                ratio = float(mixture_ratio)
            except (TypeError, ValueError) as error:
                raise InvalidConfigurationError(
                    request="mixture_ratio",
                    reason="Provide a finite nonnegative scalar mass ratio.",
                    **context,
                ) from error
            if (
                isinstance(mixture_ratio, (bool, np.bool_))
                or not np.isfinite(ratio)
                or ratio < 0
            ):
                invalid(
                    "mixture_ratio", "Provide a finite nonnegative scalar mass ratio."
                )
            mixture_ratio = ratio
            if metastable_phase is not None:
                invalid(
                    "metastable_phase",
                    "Mixture and metastable table modes cannot be combined.",
                )
        if metastable_phase is not None and gradient_method == "central":
            raise UnsupportedOptionError(
                request="gradient_method",
                reason="Metastable central differences are not implemented; choose forward differences.",
                supported=("forward",),
                **context,
            )
        if table_name is not None and (
            not isinstance(table_name, str)
            or not table_name
            or table_name in (".", "..")
            or any(c in table_name for c in "/\\:\x00")
        ):
            invalid(
                "table_name",
                "Use a file stem without directory separators or drive components.",
            )
        # store basic params
        self.fluid_name = fluid_name
        self.backend = backend
        self.h_min, self.h_max, self.N_h = h_min, h_max, N_h
        self.p_min, self.p_max, self.N_p = p_min, p_max, N_p
        self.mixture_ratio = mixture_ratio
        self.identifier = identifier
        self.provider = "bicubic"
        self.model = "binary_mixture" if mixture_ratio is not None else "real_fluid"
        self.metastable_phase = metastable_phase
        self.N_p_sat = N_p_sat
        self.table_mode = (
            "mixture"
            if is_mixture
            else (
                "equilibrium"
                if metastable_phase is None
                else f"metastable_{metastable_phase}"
            )
        )
        self.table_dir = os.fspath(table_dir)
        self.grad_method = gradient_method
        self.coarse_step = coarse_step
        self.inverse_max_steps = int(inverse_max_steps)
        self.inverse_tolerance = float(inverse_tolerance)
        self._configuration = build_configuration(
            fluid_name,
            self.provider,
            self.model,
            backend,
            identifier,
            table_options=dict(
                h_min=h_min,
                h_max=h_max,
                p_min=p_min,
                p_max=p_max,
                N_h=N_h,
                N_p=N_p,
                N_p_sat=N_p_sat,
                coarse_step=coarse_step,
                inverse_max_steps=self.inverse_max_steps,
                inverse_tolerance=self.inverse_tolerance,
                gradient_method=gradient_method,
                table_name=table_name,
                table_dir=table_dir,
                mixture_ratio=mixture_ratio,
                metastable_phase=metastable_phase,
            ),
        )
        self._capabilities = FluidCapabilities(
            input_pairs=frozenset(self.PROPERTY_CALCULATORS),
            properties=LEGACY_MIXTURE_PROPERTIES if is_mixture else SOURCE_PROPERTIES,
            jax_modes={"eager", "jit", "vmap", "jvp", "grad"},
            derivative_order=1,
            derivative_method="autodiff",
            phase_operations={"saturation"} if metastable_phase is not None else (),
            domain=FluidDomain(
                description=f"{self.table_mode} interpolation in bounded enthalpy/log-pressure coordinates.",
                bounds={"enthalpy": (h_min, h_max), "pressure": (p_min, p_max)},
                phases={self.table_mode},
                extrapolation=False,
            ),
            error_modes={"raise", "mask", "clip"},
        )

        # Create h and logP grids
        self.h_vals = jnp.linspace(self.h_min, self.h_max, self.N_h)
        self.logP_vals = jnp.linspace(np.log(self.p_min), np.log(self.p_max), self.N_p)
        self.delta_h = float(self.h_vals[1] - self.h_vals[0])
        self.delta_logP = float(self.logP_vals[1] - self.logP_vals[0])

        # Reference snapshots contain immutable scalar metadata, never static
        # FluidState arrays. Their availability is independent of table mode.
        if is_mixture:
            components = tuple(
                initialize_source_references(component, backend)
                for component in fluid_name[: -len("_mixture")].split("_")
            )
            self._references = composite_source_references(
                fluid_name,
                backend,
                components,
                mass_fractions=fractions_from_ratio(mixture_ratio),
                component_reference_ids=tuple(c.reference_id for c in components),
            )
        else:
            self._references = initialize_source_references(fluid_name, backend)
        self.reference_id = self._references.reference_id
        component_names = (
            fluid_name[: -len("_mixture")].split("_") if is_mixture else (fluid_name,)
        )
        try:
            domains = []
            for component_name in component_names:
                source = CP.AbstractState(backend, component_name)
                domains.append((source.Tmin(), source.Tmax(), source.pmax()))
            source_domain = dict(
                T_min=max(float(value[0]) for value in domains),
                T_max=min(float(value[1]) for value in domains),
                p_max=min(float(value[2]) for value in domains),
            )
        except (ValueError, RuntimeError, NotImplementedError) as error:
            raise InvalidConfigurationError(
                **context,
                request="source_domain",
                reason="The selected source must provide temperature and pressure validity limits for table generation.",
            ) from error
        critical = self._references["critical_point"]
        if critical.available and self.p_min < 0.98 * critical.properties["pressure"]:
            self.logP_sat_vals = jnp.linspace(
                np.log(self.p_min),
                np.log(critical.properties["pressure"] * 0.98),
                self.N_p_sat,
            )
            self.delta_logP_sat = float(self.logP_sat_vals[1] - self.logP_sat_vals[0])
        else:
            self.logP_sat_vals = None
            self.delta_logP_sat = None

        if metastable_phase is not None and self.logP_sat_vals is None:
            invalid(
                "saturation_domain",
                "Metastable table construction requires available source saturation data and p_min < 0.98*p_critical.",
            )
        # One immutable generation identity is shared by every writer and reader.
        # Runtime identity, inverse-guess stride and cache location are excluded.
        import jaxprop
        import jaxlib
        import scipy
        import pysolver_view

        code_files = (
            Path(__file__),
            Path(__file__).with_name("cache_metadata.py"),
            Path(__file__).with_name("table_cache.py"),
            Path(__file__).with_name("phase_envelope.py"),
            Path(__file__).parents[1] / "coolprop" / "core_calculations.py",
            Path(__file__).parents[1] / "coolprop" / "source_context.py",
            Path(__file__).parents[1] / "coolprop" / "fluid_properties.py",
            Path(__file__).parents[1] / "fluid_state.py",
            Path(__file__).parents[1] / "property_schema.py",
            Path(__file__).parents[1] / "source_references.py",
        )
        if is_mixture:
            # A thin legacy constructor now delegates generation to this shared
            # model. Fingerprint the actual dependencies, not only its wrapper,
            # so a physics/inverse change cannot reuse stale mixture tables.
            code_files += tuple(
                Path(__file__).parents[1] / "binary_mixture" / name
                for name in ("fluid.py", "host.py", "physics.py", "compat.py")
            )
        code_identity = {
            str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in code_files
        }
        self.metadata = dict(
            cache_format_version=CACHE_FORMAT_VERSION,
            generator_revision=GENERATOR_REVISION,
            generator_code_fingerprint=metadata_fingerprint(code_identity),
            jaxprop_version=jaxprop.__version__,
            jax_version=jax.__version__,
            jaxlib_version=jaxlib.__version__,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            pysolver_view_version=pysolver_view.__version__,
            fluid=self.fluid_name,
            backend=self.backend,
            table_mode=self.table_mode,
            mixture_ratio=self.mixture_ratio,
            metastable_phase=self.metastable_phase,
            gradient_method=self.grad_method,
            N_p_sat=int(self.N_p_sat),
            h_min=float(self.h_min),
            h_max=float(self.h_max),
            p_min=float(self.p_min),
            p_max=float(self.p_max),
            N_h=int(self.N_h),
            N_p=int(self.N_p),
            delta_h=self.delta_h,
            delta_logP=self.delta_logP,
            numeric_dtype=str(self.h_vals.dtype),
            jax_enable_x64=bool(jax.config.jax_enable_x64),
            jax_platform=jax.default_backend(),
            jax_default_matmul_precision=jax.config.jax_default_matmul_precision,
            properties=list(jxp.PROPERTIES_CANONICAL),
            source_signature=source_signature(
                self.fluid_name, self.backend, self.mixture_ratio
            ),
            source_domain=source_domain,
            generation_settings=dict(
                finite_difference_relative_step=1e-5,
                step_scale_floor=1.0,
                metastable_enthalpy_step_scale="table_enthalpy_span",
                metastable_enthalpy_residual_scale="cp_times_temperature_and_step",
                metastable_step_rounding_fraction=1e-3,
                phase_envelope=copy.deepcopy(PHASE_ENVELOPE_SETTINGS),
                grid_axes=["enthalpy", "log_pressure"],
                saturation_critical_pressure_fraction=0.98,
                metastable_solver_defaults={
                    name: parameter.default
                    for name, parameter in inspect.signature(
                        jxp.FluidCoolProp.get_state_metastable
                    ).parameters.items()
                    if name.startswith("solver_")
                },
                metastable_generation_solver_tolerance=1e-10,
                metastable_initial_seed="matching_homogeneous_or_requested_saturation_endpoint",
                metastable_column_seed="validated_homogeneous_anchor_row",
                metastable_residual_step_fraction=1e-8,
                metastable_residual_roundoff_ulps=512,
                metastable_mixed_derivative_scheme="central",
                metastable_mixed_step_factor=100.0,
                saturation_log_pressure_min=(
                    None if self.logP_sat_vals is None else float(self.logP_sat_vals[0])
                ),
                saturation_log_pressure_max=(
                    None
                    if self.logP_sat_vals is None
                    else float(self.logP_sat_vals[-1])
                ),
            ),
        )
        if (
            self.metadata["source_signature"].get("source_reference_id")
            != self.reference_id
        ):
            raise InvalidConfigurationError(
                **context,
                request="source_reference",
                reason="Source reference convention changed during table initialization.",
            )
        if table_name is None:
            label = "".join(c if c.isalnum() or c in "_-" else "_" for c in fluid_name)[
                :48
            ]
            self.table_name = f"{label}_{self.table_mode}_{N_h}x{N_p}_{metadata_fingerprint(self.metadata)[:20]}"
        else:
            self.table_name = table_name

        # The persisted payload includes its full generation manifest. Keep
        # only numerical entries in the dynamic tree so Equinox does not walk
        # source fingerprints and generation settings on every property call.
        table = self._load_or_generate_table()
        self._inspection = InspectionPayload(table)
        self._table, self._coefficient_validity, self._cell_validity = prepare_runtime(
            table,
            self.capabilities.properties,
            self.table_mode,
            critical.properties["pressure"] if critical.available else None,
        )
        self._input_fallbacks = FrozenMapping(
            {
                name: float(np.median(np.asarray(table[name]["value"])))
                for name in (
                    "enthalpy",
                    "pressure",
                    "temperature",
                    "density",
                    "entropy",
                )
            }
        )
        self._saturation_table, self._saturation_validity = prepare_saturation(
            table, self.capabilities.properties
        )
        self._saturation_p_max = (
            float(jnp.exp(self.logP_sat_vals[-1]))
            if self.logP_sat_vals is not None
            else self.p_max
        )
        self._saturation_capabilities = FluidCapabilities(
            input_pairs={jxp.PQ_INPUTS},
            properties=self.capabilities.properties,
            phase_operations={"saturation"} if metastable_phase is not None else (),
            jax_modes={"eager", "jit", "vmap", "jvp", "grad"},
            error_modes={"raise", "mask", "clip"},
            derivative_order=1,
            derivative_method="autodiff",
            domain=FluidDomain(
                description="The tabulated saturated liquid or vapor endpoint.",
                bounds={
                    "pressure": (
                        self.p_min,
                        (
                            float(jnp.exp(self.logP_sat_vals[-1]))
                            if self.logP_sat_vals is not None
                            else self.p_max
                        ),
                    )
                },
            ),
        )

    @property
    def name(self):
        return self.fluid_name

    @property
    def configuration(self):
        return self._configuration

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def table(self):
        """Expose the complete payload for inspection without tracing metadata.

        The outer dictionary is a snapshot; numerical entries share the stored
        arrays. Generation/reload validate the complete persisted manifest.
        """
        return inspection_view(self)

    # ------------------ Table loading / generation ------------------
    def _load_or_generate_table(self):
        cacheable = self.metadata["source_signature"].get(
            "deterministic_cache_identity", False
        )
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        if cacheable:
            os.makedirs(self.table_dir, exist_ok=True)
            try:
                with open(pkl_path, "rb") as f:
                    table = pickle.load(f)
                reason = table_validation_error(table, self.metadata)
            except FileNotFoundError:
                reason = None
                table = None
            except (
                pickle.UnpicklingError,
                EOFError,
                AttributeError,
                ImportError,
                IndexError,
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                reason = f"unreadable or malformed cache ({type(error).__name__})"
                table = None
        else:
            warnings.warn(
                "Source-file/definition identity could not be verified; persistent table caching is disabled for this source.",
                UserWarning,
                stacklevel=2,
            )
            reason = None
            table = None
        if table is not None and reason is None:
            print(f"Loaded property table from: {pkl_path}")
            return normalize_generated_arrays(table, self.metadata)
        if reason is not None:
            warnings.warn(
                f"Regenerating property table '{pkl_path}': {reason}.",
                UserWarning,
                stacklevel=2,
            )
        else:
            print("No matching table found, generating a new one...")
        generators = {
            "equilibrium": self._generate_property_table,
            "mixture": self._generate_mixture_property_table,
            "metastable_liquid": self._generate_metastable_liquid_property_table,
            "metastable_vapor": self._generate_metastable_vapor_property_table,
        }
        try:
            table = generators[self.table_mode]()
            if isinstance(table, dict):
                table["phase_cell_validity"] = self._source_phase_cell_validity()
            table = normalize_generated_arrays(table, self.metadata)
            reason = table_validation_error(table, self.metadata)
        except CalculationError:
            raise
        except Exception as error:
            raise CalculationError(
                fluid_name=self.fluid_name,
                provider="bicubic",
                model=self.model,
                request="table_generation",
                reason=f"Unable to generate a complete {self.table_mode} table: {error}",
            ) from error
        if reason is not None:
            raise CalculationError(
                fluid_name=self.fluid_name,
                provider="bicubic",
                model=self.model,
                request="table_generation",
                reason=f"Generated table was rejected: {reason}",
            )
        # Catch a reference/configuration change during construction before
        # publishing numerical data under the earlier source identity.
        if (
            source_signature(self.fluid_name, self.backend, self.mixture_ratio)
            != self.metadata["source_signature"]
        ):
            raise CalculationError(
                fluid_name=self.fluid_name,
                provider="bicubic",
                model=self.model,
                request="table_generation",
                reason="Source configuration or reference convention changed during table generation.",
            )
        if cacheable:
            self._save_table(table)
        return table

    def _save_table(self, table):
        path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        save_table_atomic(table, path)
        print(f"Saved property table to: {path}")

    def _source_phase_cell_validity(self):
        """Persist adaptive source-envelope guards without native EOS queries."""
        shape = (self.N_h - 1, self.N_p - 1)
        if self.table_mode != "equilibrium":
            return np.ones(shape, dtype=bool)
        critical = self.references["critical_point"]
        if not critical.available:
            # Ordinary calls remain usable when a source reference is absent;
            # the runtime nodal guard still applies in that configuration.
            return np.ones(shape, dtype=bool)
        if self.p_min >= critical.properties["pressure"]:
            return np.ones(shape, dtype=bool)
        source = CP.AbstractState(self.backend, self.fluid_name)
        actual_references = initialize_source_references(
            self.fluid_name, self.backend, state=source
        )
        if actual_references.reference_id != self.reference_id:
            raise CalculationError(
                fluid_name=self.fluid_name,
                provider="bicubic",
                model=self.model,
                request="table_generation",
                reason="Source reference convention changed before phase-envelope validation.",
            )
        minima = [
            point.properties["pressure"]
            for point in (
                self.references["triple_point_liquid"],
                self.references["triple_point_vapor"],
            )
            if point.available
        ]
        return source_phase_cell_validity(
            source,
            self.h_vals,
            self.logP_vals,
            critical_pressure=critical.properties["pressure"],
            critical_enthalpy=critical.properties["enthalpy"],
            minimum_saturation_pressure=min(minima) if minima else None,
            settings=self.metadata["generation_settings"]["phase_envelope"],
        )

    def _initialize_table_arrays(self):
        """Use one schema and exact independent coordinates in every mode."""
        shape = (self.N_h, self.N_p)
        table = {"metadata": copy.deepcopy(self.metadata)}
        for name in jxp.PROPERTIES_CANONICAL:
            table[name] = {
                key: np.full(shape, np.nan, dtype=np.float64)
                for key in (
                    "value",
                    "grad_h",
                    "grad_p",
                    "grad_logP",
                    "grad_ph",
                    "grad_hlogP",
                )
            }
            table[name]["coeffs"] = np.full(
                (self.N_h - 1, self.N_p - 1, 16), np.nan, dtype=np.float64
            )
        h = np.broadcast_to(np.asarray(self.h_vals)[:, None], shape).copy()
        p = np.broadcast_to(np.exp(np.asarray(self.logP_vals))[None, :], shape).copy()
        for name, values in (("enthalpy", h), ("pressure", p)):
            table[name].update(
                value=values,
                grad_h=np.ones(shape) if name == "enthalpy" else np.zeros(shape),
                grad_p=np.ones(shape) if name == "pressure" else np.zeros(shape),
                grad_logP=p.copy() if name == "pressure" else np.zeros(shape),
                grad_ph=np.zeros(shape),
                grad_hlogP=np.zeros(shape),
            )
        return table

    def _generate_property_table(self):
        # Initialize fluid (single component)
        fluid = jxp.FluidCoolProp(self.fluid_name, self.backend)

        # metadata
        table = self._initialize_table_arrays()
        enthalpy_vals = table["enthalpy"]["value"]
        pressure_vals = table["pressure"]["value"]

        keys_to_skip = {"metadata", "pressure", "enthalpy"}

        # Start property calculations
        total_points = self.N_h * self.N_p
        success_count = 0  # direct gradient evaluations
        filled_count = 0  # points actually written in table
        failure_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for i, h in enumerate(np.asarray(self.h_vals)):
                for j, logP in enumerate(np.asarray(self.logP_vals)):
                    p = float(np.exp(logP))
                    # finite difference steps (avoid zero)
                    eps_h = 1e-5 * max(abs(h), 1.0)
                    eps_p = 1e-5 * max(abs(p), 1.0)

                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_forward(fluid, h, p, eps_h, eps_p)
                        elif self.grad_method == "central":
                            grads = self._gradients_central(fluid, h, p, eps_h, eps_p)
                        else:
                            raise ValueError(
                                f"Unknown gradient scheme: {self.grad_method}"
                            )
                        success_count += 1
                    except Exception as error:
                        raise CalculationError(
                            fluid_name=self.fluid_name,
                            provider="bicubic",
                            model=self.model,
                            request="table_generation",
                            reason=f"Source evaluation failed at pressure {p!r}: {error}",
                        ) from error

                    # store computed properties
                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        if k in keys_to_skip:
                            continue
                        table[k]["value"][i, j] = float(val)
                        table[k]["grad_h"][i, j] = float(grad_h)
                        table[k]["grad_p"][i, j] = float(grad_p)
                        # store derivatives wrt logP = (d/dP) * P
                        table[k]["grad_logP"][i, j] = float(grad_p * p)
                        table[k]["grad_ph"][i, j] = float(grad_hp)
                        table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                    pbar.update(1)

            # compute coefficients after full table computed
            for k in jxp.PROPERTIES_CANONICAL:
                table[k]["coeffs"] = compute_bicubic_coefficients(
                    table[k]["value"],
                    table[k]["grad_h"],
                    table[k]["grad_logP"],
                    table[k]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

            # compute enthalpy/pressure coeffs too (they were created earlier)
            table["enthalpy"]["coeffs"] = compute_bicubic_coefficients(
                table["enthalpy"]["value"],
                table["enthalpy"]["grad_h"],
                table["enthalpy"]["grad_logP"],
                table["enthalpy"]["grad_hlogP"],
                self.delta_h,
                self.delta_logP,
            )
            table["pressure"]["coeffs"] = compute_bicubic_coefficients(
                table["pressure"]["value"],
                table["pressure"]["grad_h"],
                table["pressure"]["grad_logP"],
                table["pressure"]["grad_hlogP"],
                self.delta_h,
                self.delta_logP,
            )

        # report
        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # convert to jax arrays for all relevant sub-keys
        for k in table.keys():
            # skip metadata dictionary
            if k == "metadata":
                continue
            for sub in [
                "value",
                "grad_h",
                "grad_p",
                "grad_logP",
                "grad_ph",
                "grad_hlogP",
                "coeffs",
            ]:
                # some entries (like coeffs) might already be jnp arrays but this is safe
                table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        table["saturation_props"] = {}

        # save
        return table

    def _generate_mixture_property_table(self):
        # initialize mixture fluid object (expects mixture name / backend + ratio)\

        fluidmix = jxp.FluidMix(self.fluid_name, self.backend, self.mixture_ratio)

        table = self._initialize_table_arrays()
        enthalpy_vals = table["enthalpy"]["value"]
        pressure_vals = table["pressure"]["value"]

        keys_to_skip = {"metadata", "pressure", "enthalpy"}

        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for i, h in enumerate(np.asarray(self.h_vals)):
                for j, logP in enumerate(np.asarray(self.logP_vals)):
                    p = float(np.exp(logP))
                    eps_h = 1e-5 * max(abs(h), 1.0)
                    eps_p = 1e-5 * max(abs(p), 1.0)

                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_forward(
                                fluidmix, h, p, eps_h, eps_p
                            )
                        elif self.grad_method == "central":
                            grads = self._gradients_central(
                                fluidmix, h, p, eps_h, eps_p
                            )
                        else:
                            m = self.grad_method
                            raise ValueError(f"Unknown gradient scheme: {m}")
                        success_count += 1
                    except Exception as error:
                        raise CalculationError(
                            fluid_name=self.fluid_name,
                            provider="bicubic",
                            model=self.model,
                            request="table_generation",
                            reason=f"Source evaluation failed at pressure {p!r}: {error}",
                        ) from error

                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        if k in keys_to_skip:
                            continue
                        table[k]["value"][i, j] = val
                        table[k]["grad_h"][i, j] = grad_h
                        table[k]["grad_p"][i, j] = grad_p
                        table[k]["grad_logP"][i, j] = grad_p * p
                        table[k]["grad_ph"][i, j] = grad_hp
                        table[k]["grad_hlogP"][i, j] = grad_hp * p

                    # Update progress bar
                    pbar.update(1)

            # compute coefficients
            for k in jxp.PROPERTIES_CANONICAL:
                table[k]["coeffs"] = compute_bicubic_coefficients(
                    table[k]["value"],
                    table[k]["grad_h"],
                    table[k]["grad_logP"],
                    table[k]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # # convert to jax arrays
        # for k in table.keys():
        #     if k == "metadata":
        #         continue
        #     for sub in [
        #         "value",
        #         "grad_h",
        #         "grad_p",
        #         "grad_logP",
        #         "grad_ph",
        #         "grad_hlogP",
        #         "coeffs",
        #     ]:
        #         table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        # # convert to jax arrays
        # for k in table.keys():
        #     if k == "metadata":
        #         continue
        #     for sub in [
        #         "value",
        #         "grad_h",
        #         "grad_p",
        #         "grad_logP",
        #         "grad_ph",
        #         "grad_hlogP",
        #         "coeffs",
        #     ]:
        #         table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        # Convert all numpy arrays to jax.numpy
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_h", "grad_p", "grad_ph", "coeffs"]:
                table[k][sub] = jnp.array(table[k][sub])

        # # save
        # pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        # with open(pkl_path, "wb") as f:
        #     pickle.dump(table, f)

        table["saturation_props"] = {}

        # Save to pickle file
        return table

    def _generate_metastable_liquid_property_table(self):
        # Initialize fluid (single component)
        fluid = jxp.FluidCoolProp(self.fluid_name, self.backend)
        # The coordinate span is invariant under an enthalpy reference shift.
        # An absolute-h step would change physical derivative accuracy solely
        # because the application selected a different energy zero.
        eps_h = 1e-5 * max(self.h_max - self.h_min, 1.0)

        # metadata
        table = self._initialize_table_arrays()
        enthalpy_vals = table["enthalpy"]["value"]
        pressure_vals = table["pressure"]["value"]

        rho_guess, T_guess = self._metastable_initial_guess(
            fluid, enthalpy_vals[0][0], pressure_vals[-1][-1], "liquid"
        )

        # Define the specific keys you want to protect/skip
        keys_to_skip = {"metadata", "pressure", "enthalpy"}

        # Start evaluating properties
        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for j, logP in reversed(list(enumerate(np.asarray(self.logP_vals)))):
                p = float(np.exp(logP))
                # BUGFIX 2026-07-21: the spinodal latch must be PER PRESSURE
                # COLUMN.  As a run-global flag (old code), the first trip froze
                # every remaining point of the WHOLE table at the previous h-row
                # — the shipped water_meta_liquid_90x90 table was row h_min
                # broadcast everywhere (T 294-297 K for ALL h, zero h-variation).

                # 3. Inner Loop: Standard (Min Enthalpy -> Max Enthalpy)
                for i, h in enumerate(np.asarray(self.h_vals)):
                    # print(f"({i},{j})")
                    # finite difference steps (avoid zero)
                    eps_p = 1e-5 * max(abs(p), 1.0)

                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_metastable_forward(
                                fluid, h, p, eps_h, eps_p, rho_guess, T_guess
                            )
                            # print("grads:",grads.items())
                            # print("\n")
                            # print("table:", table.keys())
                        elif self.grad_method == "central":
                            grads = self._gradients_metastable_central(
                                fluid, h, p, eps_h, eps_p
                            )
                        else:
                            raise ValueError(
                                f"Unknown gradient scheme: {self.grad_method}"
                            )
                        success_count += 1

                    except Exception as error:
                        raise CalculationError(
                            fluid_name=self.fluid_name,
                            provider="bicubic",
                            model=self.model,
                            request="table_generation",
                            reason=f"Source evaluation failed at pressure {p!r}: {error}",
                        ) from error

                    # Update rho guess and T guess for the next grid point
                    # rho_guess, T_guess = grads["density"][0], grads["temperature"][0]

                    rho_guess = grads["density"][0] + grads["density"][1] * self.delta_h
                    T_guess = (
                        grads["temperature"][0] + grads["temperature"][1] * self.delta_h
                    )
                    # print(f"point:({i},{j}) | {grads["isothermal_compressibility"][0]}")
                    # store computed properties
                    # NOTE: We use [i, j]. Even though we are looping P backwards,
                    # 'j' is the correct index for that specific pressure in the matrix.
                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        if k in keys_to_skip:
                            continue
                        table[k]["value"][i, j] = float(val)
                        table[k]["grad_h"][i, j] = float(grad_h)
                        table[k]["grad_p"][i, j] = float(grad_p)
                        # store derivatives wrt logP = (d/dP) * P
                        table[k]["grad_logP"][i, j] = float(grad_p * p)
                        table[k]["grad_ph"][i, j] = float(grad_hp)
                        table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                    if grads["isothermal_bulk_modulus"][0] < 0:
                        raise CalculationError(
                            fluid_name=self.fluid_name,
                            provider="bicubic",
                            model=self.model,
                            request="table_generation",
                            reason="The metastable continuation routine stopped inside the requested grid; refusing to cache copied or extrapolated nodes.",
                            index=(i, j),
                        )

                        # print("After spinodal")

                    # print(f"point:{i,j}")
                    # print(f"IBM:{grads["isothermal_bulk_modulus"][0]}")
                    # print(f"IBM-ratio:{ratio}")

                    pbar.update(1)

                # Continue the requested homogeneous branch across pressure
                # columns. An ordinary HP flash at this anchor can be an
                # equilibrium mixture, whose bulk density is not a liquid
                # root guess, even though a valid metastable liquid exists.
                rho_guess = table["density"]["value"][0, j]
                T_guess = table["temperature"]["value"][0, j]

                # compute coefficients after full table computed
                for k in jxp.PROPERTIES_CANONICAL:
                    table[k]["coeffs"] = compute_bicubic_coefficients(
                        table[k]["value"],
                        table[k]["grad_h"],
                        table[k]["grad_logP"],
                        table[k]["grad_hlogP"],
                        self.delta_h,
                        self.delta_logP,
                    )

                # compute enthalpy/pressure coeffs too (they were created earlier)
                table["enthalpy"]["coeffs"] = compute_bicubic_coefficients(
                    table["enthalpy"]["value"],
                    table["enthalpy"]["grad_h"],
                    table["enthalpy"]["grad_logP"],
                    table["enthalpy"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )
                table["pressure"]["coeffs"] = compute_bicubic_coefficients(
                    table["pressure"]["value"],
                    table["pressure"]["grad_h"],
                    table["pressure"]["grad_logP"],
                    table["pressure"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        # report
        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        # print(
        #     f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        # )
        # print(f"Total table generation time: {elapsed:.2f} s")

        # convert to jax arrays for all relevant sub-keys
        for k in table.keys():
            # skip metadata dictionary
            if k == "metadata":
                continue
            for sub in [
                "value",
                "grad_h",
                "grad_p",
                "grad_logP",
                "grad_ph",
                "grad_hlogP",
                "coeffs",
            ]:
                # some entries (like coeffs) might already be jnp arrays but this is safe
                table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        print("Generating saturation properties table...")
        table["saturation_props"] = {}
        for k in jxp.PROPERTIES_CANONICAL:
            table["saturation_props"][k] = {
                "value": np.full(self.N_p_sat, np.nan, dtype=np.float64),
                "grad_p": np.full(self.N_p_sat, np.nan, dtype=np.float64),
            }

        # Loop from pmin to 0.99 p_crit
        # store sat props using gradients_saturation_forward and gradients_saturation_central
        # gradients_saturation_forward and gradients_saturation_central will use jxp.PQ_inputs of coolprop fluid object
        # calculate grad_p for calculation of coefficients

        for j, logP in list(enumerate(np.asarray(self.logP_sat_vals))):
            p = float(np.exp(logP))
            eps_p = 1e-5 * max(abs(p), 1.0)
            try:
                if self.grad_method == "forward":
                    grads = self._gradients_saturation_forward(fluid, p, 0.0, eps_p)
                elif self.grad_method == "central":
                    grads = self._gradients_saturation_central(fluid, p, 0.0, eps_p)
                else:
                    raise ValueError(f"Unknown gradient scheme: {self.grad_method}")

            except Exception as error:
                raise CalculationError(
                    fluid_name=self.fluid_name,
                    provider="bicubic",
                    model=self.model,
                    request="table_generation",
                    reason=f"Source evaluation failed at pressure {p!r}: {error}",
                ) from error

            # store computed properties
            for k, (val, grad_p) in grads.items():
                table["saturation_props"][k]["value"][j] = float(val)
                table["saturation_props"][k]["grad_p"][j] = float(grad_p)

        # Convert saturation props to JAX arrays for JIT/vmap compatibility
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_p"]:
                table["saturation_props"][k][sub] = jnp.array(
                    table["saturation_props"][k][sub], dtype=jnp.float64
                )

        elapsed = time.perf_counter() - start_time
        # Printing outcome
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # save
        return table

    def _generate_metastable_vapor_property_table(self):
        # Initialize fluid (single component)
        fluid = jxp.FluidCoolProp(self.fluid_name, self.backend)
        eps_h = 1e-5 * max(self.h_max - self.h_min, 1.0)

        # metadata
        table = self._initialize_table_arrays()
        enthalpy_vals = table["enthalpy"]["value"]
        pressure_vals = table["pressure"]["value"]

        rho_guess, T_guess = self._metastable_initial_guess(
            fluid, enthalpy_vals[-1][-1], pressure_vals[-1][-1], "vapor"
        )

        # Define the specific keys you want to protect/skip
        keys_to_skip = {"metadata", "pressure", "enthalpy"}

        # Start evaluating properties
        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for j, logP in reversed(list(enumerate(np.asarray(self.logP_vals)))):
                p = float(np.exp(logP))
                # BUGFIX 2026-07-21: per-column spinodal latch (see the liquid
                # generator) — a run-global flag freezes the whole remaining
                # table after the first trip.

                # 3. Inner Loop: Standard (Min Enthalpy -> Max Enthalpy)
                for i, h in reversed(list(enumerate(np.asarray(self.h_vals)))):
                    # print(f"({i},{j})")
                    # finite difference steps (avoid zero)
                    eps_p = 1e-5 * max(abs(p), 1.0)

                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_metastable_forward(
                                fluid, h, p, eps_h, eps_p, rho_guess, T_guess
                            )
                            # print("grads:",grads.items())
                            # print("\n")
                            # print("table:", table.keys())
                        elif self.grad_method == "central":
                            grads = self._gradients_metastable_central(
                                fluid, h, p, eps_h, eps_p
                            )
                        else:
                            raise ValueError(
                                f"Unknown gradient scheme: {self.grad_method}"
                            )
                        success_count += 1

                    except Exception as error:
                        raise CalculationError(
                            fluid_name=self.fluid_name,
                            provider="bicubic",
                            model=self.model,
                            request="table_generation",
                            reason=f"Source evaluation failed at pressure {p!r}: {error}",
                        ) from error

                    # Update rho guess and T guess for the next grid point
                    # rho_guess, T_guess = grads["density"][0], grads["temperature"][0]

                    rho_guess = grads["density"][0] - grads["density"][1] * self.delta_h
                    T_guess = (
                        grads["temperature"][0] - grads["temperature"][1] * self.delta_h
                    )
                    # print(f"point:({i},{j}) | {grads["isothermal_compressibility"][0]}")
                    # store computed properties
                    # NOTE: We use [i, j]. Even though we are looping P backwards,
                    # 'j' is the correct index for that specific pressure in the matrix.
                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        if k in keys_to_skip:
                            continue
                        table[k]["value"][i, j] = float(val)
                        table[k]["grad_h"][i, j] = float(grad_h)
                        table[k]["grad_p"][i, j] = float(grad_p)
                        # store derivatives wrt logP = (d/dP) * P
                        table[k]["grad_logP"][i, j] = float(grad_p * p)
                        table[k]["grad_ph"][i, j] = float(grad_hp)
                        table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                    if grads["isothermal_bulk_modulus"][0] < 0:
                        raise CalculationError(
                            fluid_name=self.fluid_name,
                            provider="bicubic",
                            model=self.model,
                            request="table_generation",
                            reason="The metastable continuation routine stopped inside the requested grid; refusing to cache copied or extrapolated nodes.",
                            index=(i, j),
                        )

                        # print("After spinodal")

                    # print(f"point:{i,j}")
                    # print(f"IBM:{grads["isothermal_bulk_modulus"][0]}")
                    # print(f"IBM-ratio:{ratio}")

                    pbar.update(1)

                # The last enthalpy row is this vapor sweep's validated
                # homogeneous anchor, not an ordinary two-phase bulk state.
                rho_guess = table["density"]["value"][-1, j]
                T_guess = table["temperature"]["value"][-1, j]

                # compute coefficients after full table computed
                for k in jxp.PROPERTIES_CANONICAL:
                    table[k]["coeffs"] = compute_bicubic_coefficients(
                        table[k]["value"],
                        table[k]["grad_h"],
                        table[k]["grad_logP"],
                        table[k]["grad_hlogP"],
                        self.delta_h,
                        self.delta_logP,
                    )

                # compute enthalpy/pressure coeffs too (they were created earlier)
                table["enthalpy"]["coeffs"] = compute_bicubic_coefficients(
                    table["enthalpy"]["value"],
                    table["enthalpy"]["grad_h"],
                    table["enthalpy"]["grad_logP"],
                    table["enthalpy"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )
                table["pressure"]["coeffs"] = compute_bicubic_coefficients(
                    table["pressure"]["value"],
                    table["pressure"]["grad_h"],
                    table["pressure"]["grad_logP"],
                    table["pressure"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        # report
        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        # print(
        #     f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        # )
        # print(f"Total table generation time: {elapsed:.2f} s")

        # convert to jax arrays for all relevant sub-keys
        for k in table.keys():
            # skip metadata dictionary
            if k == "metadata":
                continue
            for sub in [
                "value",
                "grad_h",
                "grad_p",
                "grad_logP",
                "grad_ph",
                "grad_hlogP",
                "coeffs",
            ]:
                # some entries (like coeffs) might already be jnp arrays but this is safe
                table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        print("Generating saturation properties table...")

        table["saturation_props"] = {}
        for k in jxp.PROPERTIES_CANONICAL:
            table["saturation_props"][k] = {
                "value": np.full(self.N_p_sat, np.nan, dtype=np.float64),
                "grad_p": np.full(self.N_p_sat, np.nan, dtype=np.float64),
            }

        # Loop from pmin to 0.99 p_crit
        # store sat props using gradients_saturation_forward and gradients_saturation_central
        # gradients_saturation_forward and gradients_saturation_central will use jxp.PQ_inputs of coolprop fluid object
        # calculate grad_p for calculation of coefficients

        for j, logP in list(enumerate(np.asarray(self.logP_sat_vals))):
            p = float(np.exp(logP))
            eps_p = 1e-5 * max(abs(p), 1.0)
            try:
                if self.grad_method == "forward":
                    grads = self._gradients_saturation_forward(fluid, p, 1.0, eps_p)
                elif self.grad_method == "central":
                    grads = self._gradients_saturation_central(fluid, p, 1.0, eps_p)
                else:
                    raise ValueError(f"Unknown gradient scheme: {self.grad_method}")

            except Exception as error:
                raise CalculationError(
                    fluid_name=self.fluid_name,
                    provider="bicubic",
                    model=self.model,
                    request="table_generation",
                    reason=f"Source evaluation failed at pressure {p!r}: {error}",
                ) from error

            # store computed properties
            for k, (val, grad_p) in grads.items():
                table["saturation_props"][k]["value"][j] = float(val)
                table["saturation_props"][k]["grad_p"][j] = float(grad_p)

        # Convert saturation props to JAX arrays for JIT/vmap compatibility
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_p"]:
                table["saturation_props"][k][sub] = jnp.array(
                    table["saturation_props"][k][sub], dtype=jnp.float64
                )

        elapsed = time.perf_counter() - start_time
        # Printing generation results
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # save
        return table

    # ------------------ Gradient helpers ------------------
    @staticmethod
    def _metastable_initial_guess(fluid, enthalpy, pressure, phase):
        """Select a homogeneous seed for the requested construction branch.

        Ordinary HP states are useful when they already lie on that branch or
        above the critical pressure. A subcritical equilibrium mixture (or the
        other homogeneous phase) must instead start at the requested bubble or
        dew endpoint. Subsequent columns reuse validated homogeneous anchors.
        All source queries here belong to table construction only.
        """
        state = fluid.get_state(jxp.HmassP_INPUTS, enthalpy, pressure)
        source = fluid.abstract_state
        requested = (
            (CP.iphase_liquid, CP.iphase_supercritical_liquid)
            if phase == "liquid"
            else (CP.iphase_gas, CP.iphase_supercritical_gas)
        )
        if pressure < source.p_critical() and source.phase() not in requested:
            quality = 0.0 if phase == "liquid" else 1.0
            state = fluid.get_state(jxp.PQ_INPUTS, pressure, quality)
        return float(state.rho), float(state.T)

    @staticmethod
    def _gradients_forward(fluid, h, p, eps_h, eps_p):
        # Export raw values once per sample. Optional NaNs remain available to
        # table generation without repeated bound-method lookups per property.
        f0 = fluid.get_state(jxp.HmassP_INPUTS, h, p).to_dict()
        fh = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p).to_dict()
        fp = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p).to_dict()
        fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p).to_dict()

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0.get(k, np.nan)
            grad_h = (fh.get(k, np.nan) - f0.get(k, np.nan)) / eps_h
            grad_p = (fp.get(k, np.nan) - f0.get(k, np.nan)) / eps_p
            grad_hp = (
                fhp.get(k, np.nan)
                - fh.get(k, np.nan)
                - fp.get(k, np.nan)
                + f0.get(k, np.nan)
            ) / (eps_h * eps_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)
        return grads

    @staticmethod
    def _gradients_central(fluid, h, p, eps_h, eps_p):
        f0 = fluid.get_state(jxp.HmassP_INPUTS, h, p).to_dict()
        fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p).to_dict()
        fhm = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p).to_dict()
        fph = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p).to_dict()
        fpm = fluid.get_state(jxp.HmassP_INPUTS, h, p - eps_p).to_dict()
        fh_p = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p).to_dict()
        fh_m = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p - eps_p).to_dict()
        fm_p = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p + eps_p).to_dict()
        fm_m = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p - eps_p).to_dict()

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0.get(k, np.nan)
            grad_h = (fhp.get(k, np.nan) - fhm.get(k, np.nan)) / (2.0 * eps_h)
            grad_p = (fph.get(k, np.nan) - fpm.get(k, np.nan)) / (2.0 * eps_p)
            grad_hp = (
                fh_p.get(k, np.nan)
                - fh_m.get(k, np.nan)
                - fm_p.get(k, np.nan)
                + fm_m.get(k, np.nan)
            ) / (4.0 * eps_h * eps_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)
        return grads

    @staticmethod
    def _gradients_metastable_forward(fluid, h, p, eps_h, eps_p, rho_guess, T_guess):
        """Sample the homogeneous branch with residuals below the FD budget."""
        for name, coordinate, step in (
            ("enthalpy", h, eps_h),
            ("pressure", p, eps_p),
        ):
            coordinate, step = float(coordinate), float(step)
            increments = np.asarray(
                [(coordinate + step) - coordinate, coordinate - (coordinate - step)]
            )
            if (
                not np.isfinite(step)
                or step <= 0
                or not np.all(np.isfinite(increments))
                or np.any(np.abs(increments - step) > 1e-3 * step)
            ):
                raise CalculationError(
                    fluid_name=fluid.name,
                    provider="bicubic",
                    model="real_fluid",
                    request="table_generation",
                    reason=(
                        f"The metastable {name} stencil step {step!r} cannot be "
                        f"represented accurately at coordinate {coordinate!r}. "
                        "Use a resolvable table span/reference convention; "
                        "refusing reference-dependent or zero derivatives."
                    ),
                )
        rhoT_guess = np.array([rho_guess, T_guess])
        sample = lambda enthalpy, pressure: FluidBicubic._metastable_generation_sample(
            fluid, enthalpy, pressure, eps_h, eps_p, rhoT_guess
        )
        f0 = sample(h, p)
        fh = sample(h + eps_h, p)
        fp = sample(h, p + eps_p)
        # A small forward mixed stencil subtracts nearly equal flash results.
        # Use a wider, centered mixed stencil while retaining the requested
        # forward first derivatives. This work is paid only during generation.
        mixed_h, mixed_p = 100 * eps_h, 100 * eps_p
        mixed_pp = sample(h + mixed_h, p + mixed_p)
        mixed_pm = sample(h + mixed_h, p - mixed_p)
        mixed_mp = sample(h - mixed_h, p + mixed_p)
        mixed_mm = sample(h - mixed_h, p - mixed_p)

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0.get(k, np.nan)
            grad_h = (fh.get(k, np.nan) - f0.get(k, np.nan)) / eps_h
            grad_p = (fp.get(k, np.nan) - f0.get(k, np.nan)) / eps_p
            grad_hp = (
                mixed_pp.get(k, np.nan)
                - mixed_pm.get(k, np.nan)
                - mixed_mp.get(k, np.nan)
                + mixed_mm.get(k, np.nan)
            ) / (4 * mixed_h * mixed_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)

        # print("grads:",grads.items())
        # print("\n")

        return grads

    @staticmethod
    def _metastable_generation_sample(fluid, h, p, eps_h, eps_p, rhoT_guess):
        """Validate domain and solve accuracy only during table generation."""
        properties = fluid.get_state_metastable(
            prop_1="p",
            prop_1_value=p,
            prop_2="h",
            prop_2_value=h,
            rhoT_guess=rhoT_guess,
            solver_tolerance=1e-10,
        ).to_dict()
        temperature = float(properties["temperature"])
        pressure = float(properties["pressure"])
        density = float(properties["density"])
        boundary_precision = 1e-9
        if (
            not np.isfinite(temperature)
            or not np.isfinite(pressure)
            or not np.isfinite(density)
            or density <= 0
            or pressure <= 0
            or temperature < fluid.T_min * (1 - boundary_precision)
            or temperature > fluid.T_max * (1 + boundary_precision)
            or pressure > fluid.p_max * (1 + boundary_precision)
        ):
            raise CalculationError(
                fluid_name=fluid.name,
                provider="bicubic",
                model="real_fluid",
                request="table_generation",
                reason=(
                    "The metastable sample exceeds the source EOS domain: "
                    f"temperature={temperature} K, pressure={pressure} Pa, "
                    f"density={density} kg/m^3; published temperature "
                    f"[{fluid.T_min}, {fluid.T_max}] K and pressure "
                    f"(0, {fluid.p_max}] Pa."
                ),
            )
        for name, target, step in (("enthalpy", h, eps_h), ("pressure", p, eps_p)):
            value = float(properties[name])
            # Keep coordinate errors negligible relative to the stencil while
            # allowing the unavoidable roundoff of a floating EOS evaluation.
            # In stiff liquids pressure roundoff scales with the bulk modulus,
            # not the much smaller net pressure obtained by cancellation.
            scales = (
                (
                    properties.get("isothermal_bulk_modulus", 0.0),
                    properties.get("isentropic_bulk_modulus", 0.0),
                )
                if name == "pressure"
                else (properties.get("isobaric_heat_capacity", 0.0) * temperature,)
            )
            scale = max(
                abs(target) if name == "pressure" else abs(step),
                1.0,
                *(abs(float(item)) for item in scales if np.isfinite(item)),
            )
            budget = max(
                1e-8 * abs(step),
                512 * np.finfo(float).eps * scale,
                # Offsetting an energy zero changes representable spacing,
                # not the physical residual scale. Account for that rounding
                # separately rather than granting a relative-to-offset error.
                4 * abs(np.spacing(target)),
            )
            if not np.isfinite(value) or abs(value - target) > budget:
                raise CalculationError(
                    fluid_name=fluid.name,
                    provider="bicubic",
                    model="real_fluid",
                    request="table_generation",
                    reason=(
                        f"The metastable {name} residual {value - target!r} "
                        f"exceeds the finite-difference error budget {budget!r} "
                        f"for step {step!r}; refusing inaccurate table derivatives."
                    ),
                )
        return properties

    @staticmethod
    def _gradients_saturation_forward(fluid, p, Q, eps_p):
        f0 = fluid.get_state(jxp.PQ_INPUTS, p, Q).to_dict()
        fp = fluid.get_state(jxp.PQ_INPUTS, p + eps_p, Q).to_dict()

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0.get(k, np.nan)
            grad_p = (fp.get(k, np.nan) - f0.get(k, np.nan)) / eps_p
            grads[k] = (val, grad_p)
        return grads

    @staticmethod
    def _gradients_saturation_central(fluid, p, Q, eps_p):
        f0 = fluid.get_state(jxp.PQ_INPUTS, p, Q).to_dict()
        fp_plus = fluid.get_state(jxp.PQ_INPUTS, p + eps_p, Q).to_dict()
        fp_minus = fluid.get_state(jxp.PQ_INPUTS, p - eps_p, Q).to_dict()

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0.get(k, np.nan)
            grad_p = (fp_plus.get(k, np.nan) - fp_minus.get(k, np.nan)) / (2 * eps_p)
            grads[k] = (val, grad_p)
        return grads

    # ------------------ Bicubic property interpolation ------------------
    PROPERTY_CALCULATORS = {
        pair: (
            lambda self, first, second, _pair=pair: self.get_state(_pair, first, second)
        )
        for pair in (
            jxp.HmassP_INPUTS,
            jxp.PT_INPUTS,
            jxp.DmassP_INPUTS,
            jxp.HmassSmass_INPUTS,
            jxp.DmassHmass_INPUTS,
            jxp.DmassT_INPUTS,
            jxp.DmassSmass_INPUTS,
            jxp.PSmass_INPUTS,
        )
    }

    @checked_bicubic_jit
    def get_state(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
    ):
        """Evaluate a bounded table state with explicit numerical failure policy.

        The input-pair selector is static for JAX compilation; numerical inputs
        and table coefficients remain dynamic. Each pair has its own cached
        specialization, so the same fluid object supports different advertised
        pairs throughout a simulation without specializing on numerical values.

        Eager calls default to indexed raising. Within JAX transformations,
        explicitly select ``errors='mask'`` or ``errors='clip'``.
        """
        return evaluate_state(self, input_type, prop_1, prop_2, errors=errors)

    @checked_saturation_jit
    def get_state_saturation(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
    ):
        """Evaluate the stored PQ saturation endpoint; other tables reject it."""
        return evaluate_saturation(self, prop_1, prop_2, errors=errors)

    def _get_state_scalar(self, input_type, val1, val2, *, errors="raise"):
        return evaluate_scalar(self, input_type, val1, val2, errors=errors)

    def _interp_h_p(self, h, p):
        return interpolate_properties(self, h, p)

    def _interp_sat_p(self, p):
        coordinate = (jnp.log(p) - jnp.log(self.p_min)) / self.delta_logP_sat
        index = jnp.clip(jnp.floor(coordinate).astype(jnp.int32), 0, self.N_p_sat - 2)
        pressure = self._saturation_table["pressure"]["value"]
        left, right = pressure[index], pressure[index + 1]
        width = right - left
        position = (p - left) / width
        one_minus = 1 - position
        properties = {}
        for name in self.capabilities.properties:
            entry = self._saturation_table[name]
            a, b = entry["value"][index], entry["value"][index + 1]
            ga, gb = entry["grad_p"][index], entry["grad_p"][index + 1]
            delta = b - a
            # Symmetric cubic Hermite interpolation, with t = (p-left)/width:
            # q(t) = (1-t)*a + t*b + t*(1-t) *
            #        [(1-t)*(ga*width-delta) + t*(delta-gb*width)].
            # a,b are endpoint values; ga,gb are their pressure derivatives.
            correction = (
                position
                * one_minus
                * (one_minus * (ga * width - delta) + position * (delta - gb * width))
            )
            value = one_minus * a + position * b + correction
            mask = self._saturation_validity[name]
            available = True if mask is None else mask[index]
            properties[name] = jnp.where(
                available & jnp.isfinite(value),
                value,
                jnp.nan,
            )
        properties["pressure"] = p
        properties["quality_mass"] = jnp.asarray(
            0.0 if self.metastable_phase == "liquid" else 1.0, dtype=p.dtype
        )
        return properties


# ------------------------------------------------------------------
# Bicubic coefficient computation function (unchanged logic)
def compute_bicubic_coefficients(
    value, grad_h, grad_logP, grad_hlogP, delta_h, delta_logP
):
    XX = np.stack(
        [
            value[:-1, :-1],
            value[1:, :-1],
            value[:-1, 1:],
            value[1:, 1:],
            grad_h[:-1, :-1] * delta_h,
            grad_h[1:, :-1] * delta_h,
            grad_h[:-1, 1:] * delta_h,
            grad_h[1:, 1:] * delta_h,
            grad_logP[:-1, :-1] * delta_logP,
            grad_logP[1:, :-1] * delta_logP,
            grad_logP[:-1, 1:] * delta_logP,
            grad_logP[1:, 1:] * delta_logP,
            grad_hlogP[:-1, :-1] * delta_h * delta_logP,
            grad_hlogP[1:, :-1] * delta_h * delta_logP,
            grad_hlogP[:-1, 1:] * delta_h * delta_logP,
            grad_hlogP[1:, 1:] * delta_h * delta_logP,
        ],
        axis=-1,
    )
    coeffs = np.einsum("ab,ijb->ija", A_MAT, XX)
    return coeffs


# fmt: off
# Bicubic interpolation matrix [TODO provide reference in docs]
# https://en.wikipedia.org/wiki/Bicubic_interpolation
A_MAT = np.array([
    [+1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [-3.0, +3.0, +0.0, +0.0, -2.0, -1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+2.0, -2.0, +0.0, +0.0, +1.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, -3.0, +3.0, +0.0, +0.0, -2.0, -1.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +2.0, -2.0, +0.0, +0.0, +1.0, +1.0, +0.0, +0.0],
    [-3.0, +0.0, +3.0, +0.0, +0.0, +0.0, +0.0, +0.0, -2.0, +0.0, -1.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, -3.0, +0.0, +3.0, +0.0, +0.0, +0.0, +0.0, +0.0, -2.0, +0.0, -1.0, +0.0],
    [+9.0, -9.0, -9.0, +9.0, +6.0, +3.0, -6.0, -3.0, +6.0, -6.0, +3.0, -3.0, +4.0, +2.0, +2.0, +1.0],
    [-6.0, +6.0, +6.0, -6.0, -3.0, -3.0, +3.0, +3.0, -4.0, +4.0, -2.0, +2.0, -2.0, -2.0, -1.0, -1.0],
    [+2.0, +0.0, -2.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +2.0, +0.0, -2.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +1.0, +0.0],
    [-6.0, +6.0, +6.0, -6.0, -4.0, -2.0, +4.0, +2.0, -3.0, +3.0, -3.0, +3.0, -2.0, -1.0, -2.0, -1.0],
    [+4.0, -4.0, -4.0, +4.0, +2.0, +2.0, -2.0, -2.0, +2.0, -2.0, +2.0, -2.0, +1.0, +1.0, +1.0, +1.0],
], dtype=np.float64)
# fmt: on
