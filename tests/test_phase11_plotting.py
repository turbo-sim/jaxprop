"""Headless source-EOS diagrams update artists and respect stream identity."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.coolprop import fluid_properties as source_module


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def source():
    return jp.Fluid("CO2", identifier="source-stream")


@pytest.fixture(scope="module")
def table_options(tmp_path_factory):
    return dict(
        h_min=600e3,
        h_max=850e3,
        p_min=5e6,
        p_max=12e6,
        N_h=4,
        N_p=4,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(tmp_path_factory.mktemp("phase11_plotting")),
    )


def draw(source, **options):
    return jp.plot_phase_diagram(source, N=4, **options)


def test_source_plot_creates_updates_and_hides_existing_artists(source):
    figure, axes = draw(source)
    saturation = source.graphic_elements[axes]["saturation_line"]
    critical = source.graphic_elements[axes]["critical_point"]
    initial_count = len(axes.lines)
    same_figure, same_axes = draw(source, axes=axes, show_in_legend=True)
    assert same_figure is figure and same_axes is axes
    assert len(axes.lines) == initial_count
    assert source.graphic_elements[axes]["saturation_line"] is saturation
    assert saturation.get_label() == "Saturation line"
    assert critical.get_label() == "Critical point"
    draw(source, axes=axes, plot_saturation_line=False, plot_critical_point=False)
    assert not saturation.get_visible() and not critical.get_visible()
    draw(source, axes=axes)
    assert saturation.get_visible() and critical.get_visible()
    assert source.sat_liq.identifier == source.identifier
    assert source.sat_vap.identifier == source.identifier
    figure.canvas.draw()


@pytest.mark.parametrize("method", ["coolprop", "coolprop_jax"])
def test_restored_source_reuses_numerical_curves_and_creates_fresh_artists(method):
    import pickle

    fluid = jp.Fluid("CO2", method=method, identifier="restored-stream")
    _, original_axes = draw(
        fluid,
        plot_quality_isolines=True,
        quality_levels=[0.25, 0.75],
        plot_two_phase_patch=True,
    )
    original = fluid if method == "coolprop" else fluid.fluid
    assert hasattr(original, "_graphic_saturation_line")
    assert hasattr(original, "_graphics_q_lines")
    assert not any(name.startswith("_graphic") for name in original.__getstate__())
    restored = pickle.loads(pickle.dumps(fluid))
    source = restored if method == "coolprop" else restored.fluid
    assert not source.graphic_elements
    assert not any(name.startswith("_graphic") for name in source.__dict__)
    assert source.sat_liq.identifier == "restored-stream"
    figure, axes = draw(restored)
    assert axes is not original_axes
    assert (
        source.graphic_elements[axes]["saturation_line"]
        is not original.graphic_elements[original_axes]["saturation_line"]
    )
    np.testing.assert_allclose(source.sat_liq.s, original.sat_liq.s)
    assert source.sat_liq.reference_id == original.sat_liq.reference_id
    figure.canvas.draw()


def test_sampling_and_near_critical_options_invalidate_curve_cache(source, monkeypatch):
    original = source_module.compute_saturation_line
    calls = []

    def observed(fluid, N, dT_crit):
        calls.append((N, dT_crit))
        return original(fluid, N, dT_crit=dT_crit)

    monkeypatch.setattr(source_module, "compute_saturation_line", observed)
    _, axes = source.plot_phase_diagram(N=3, dT_crit=1.0)
    source.plot_phase_diagram(axes=axes, N=3, dT_crit=1.0)
    source.plot_phase_diagram(axes=axes, N=5, dT_crit=1.0)
    assert source.sat_liq.shape == (6,)
    source.plot_phase_diagram(axes=axes, N=5, dT_crit=2.0)
    assert calls == [(3, 1.0), (5, 1.0), (5, 2.0)]
    assert len(axes.lines) == 2


def test_spinodal_solver_options_and_styles_update(source, monkeypatch):
    calls = []

    def observed(fluid, **options):
        calls.append(options)
        # Numerical spinodal roots have their independent physics tests; this
        # test isolates plotting cache keys and Matplotlib artist updates.
        return source_module.compute_saturation_line(
            fluid, options["N"], dT_crit=options["dT_crit"]
        )

    monkeypatch.setattr(source_module, "compute_spinodal_line", observed)
    _, axes = draw(source, plot_spinodal_line=True, spinodal_line_width=1.0)
    line = source.graphic_elements[axes]["spinodal_line"]
    draw(
        source,
        axes=axes,
        plot_spinodal_line=True,
        spinodal_line_width=3.0,
        spinodal_line_color="red",
        show_in_legend=True,
    )
    assert len(calls) == 1
    assert line.get_linewidth() == 3.0 and line.get_color() == "red"
    assert line.get_label() == "Spinodal line"
    draw(
        source,
        axes=axes,
        plot_spinodal_line=True,
        spinodal_line_method="bfgs",
        spinodal_line_use_previous=True,
    )
    assert len(calls) == 2
    assert calls[-1]["method"] == "bfgs"
    assert calls[-1]["use_previous_as_initial_guess"] is True


def test_supplied_axes_scales_are_applied_without_replacing_custom_labels(source):
    figure, axes = plt.subplots()
    axes.set_xlabel("my pressure")
    source.plot_phase_diagram("p", "T", axes=axes, N=4, x_scale="log", y_scale="log")
    assert axes.get_xscale() == axes.get_yscale() == "log"
    assert axes.get_xlabel() == "my pressure" and axes.get_ylabel()
    figure.canvas.draw()


def test_changing_coordinates_refreshes_generated_labels_and_limits(source):
    figure, axes = draw(source)
    old_label, old_limits = axes.get_xlabel(), axes.get_xlim()
    draw(source, axes=axes, x_prop="p", x_scale="log")
    assert axes.get_xlabel() != old_label
    assert axes.get_xlim() != old_limits
    pressure = np.asarray(source.graphic_elements[axes]["saturation_line"].get_xdata())
    low, high = axes.get_xlim()
    assert low < np.min(pressure) <= np.max(pressure) < high
    figure.canvas.draw()


def test_artist_updates_preserve_user_labels_and_manual_limits(source):
    _, axes = draw(source)
    axes.set_xlabel("custom horizontal coordinate")
    axes.set_ylabel("custom vertical coordinate")
    axes.set_xlim(100.0, 200.0)
    axes.set_ylim(250.0, 280.0)
    draw(source, axes=axes, x_prop="h")
    assert axes.get_xlabel() == "custom horizontal coordinate"
    assert axes.get_ylabel() == "custom vertical coordinate"
    np.testing.assert_array_equal(axes.get_xlim(), [100.0, 200.0])
    np.testing.assert_array_equal(axes.get_ylim(), [250.0, 280.0])


def test_two_phase_patch_is_independent_of_outline_and_reused(source):
    figure, axes = draw(source, plot_saturation_line=False, plot_two_phase_patch=True)
    patch = source.graphic_elements[axes]["two_phase_patch"]
    assert len(axes.patches) == 1 and patch.get_visible()
    draw(source, axes=axes, plot_saturation_line=False, plot_two_phase_patch=True)
    assert len(axes.patches) == 1
    assert source.graphic_elements[axes]["two_phase_patch"] is patch
    draw(source, axes=axes, plot_saturation_line=False, plot_two_phase_patch=False)
    assert not patch.get_visible()
    figure.canvas.draw()


def test_quality_levels_refresh_and_labels_do_not_accumulate(source):
    figure, axes = draw(
        source,
        plot_quality_isolines=True,
        quality_levels=[0.2, 0.6],
        quality_labels=True,
    )
    first = source.graphic_elements[axes]["quality_isolines"]
    draw(
        source,
        axes=axes,
        plot_quality_isolines=True,
        quality_levels=[0.3, 0.5, 0.7],
        quality_labels=True,
    )
    second = source.graphic_elements[axes]["quality_isolines"]
    assert first is not second
    assert first.axes is None
    np.testing.assert_allclose(second.levels, [0.3, 0.5, 0.7])
    np.testing.assert_allclose(np.asarray(source.q_mesh.Q)[:, 0], [0.3, 0.5, 0.7])
    assert len(axes.texts) == len(second.labelTexts)
    draw(source, axes=axes, plot_quality_isolines=False)
    assert "quality_isolines" not in source.graphic_elements[axes]
    assert not axes.texts
    figure.canvas.draw()


def test_single_quality_isoline_is_supported(source):
    figure, axes = draw(source, plot_quality_isolines=True, quality_levels=[0.5])
    contour = source.graphic_elements[axes]["quality_isolines"]
    np.testing.assert_allclose(contour.levels, [0.5])
    assert any(len(segment) > 0 for segment in contour.allsegs[0])
    figure.canvas.draw()


@pytest.mark.parametrize("levels", [[0.6, 0.2], [0.2, 0.2], [np.nan], [-0.1], []])
def test_invalid_quality_levels_have_explicit_configuration_error(source, levels):
    with pytest.raises(jp.InvalidConfigurationError, match="quality_levels"):
        draw(source, plot_quality_isolines=True, quality_levels=levels)


def test_pseudocritical_sampling_is_honored(source):
    _, axes = source.plot_phase_diagram(N=3, plot_pseudocritical_line=True)
    assert source.pseudo_critical_line.shape == (3,)
    source.plot_phase_diagram(axes=axes, N=6, plot_pseudocritical_line=True)
    assert source.pseudo_critical_line.shape == (6,)
    assert len(axes.lines) == 3


@pytest.mark.parametrize("method", ["coolprop", "coolprop_jax", "bicubic"])
def test_common_plotting_path_draws_source_eos_for_real_fluid_evaluators(
    method, table_options
):
    fluid = jp.Fluid(
        "CO2",
        method=method,
        identifier="common-stream",
        **({"table_options": table_options} if method == "bicubic" else {})
    )
    figure, axes = draw(fluid)
    source = (
        fluid
        if method == "coolprop"
        else (
            fluid.fluid
            if method == "coolprop_jax"
            else next(iter(axes._jaxprop_phase_sources.values()))
        )
    )
    assert source.identifier == fluid.identifier
    assert source.references.reference_id == fluid.references.reference_id
    assert source.sat_liq.identifier == fluid.identifier
    assert (
        np.max(np.asarray(source.sat_liq.h)) < 600e3
    )  # explicit source-EOS scope, outside this table
    draw(fluid, axes=axes, show_in_legend=True)
    assert len(axes.lines) == 2
    figure.canvas.draw()


def test_streams_and_axes_keep_separate_plotting_artists(table_options):
    first = jp.Fluid(
        "CO2", method="bicubic", identifier="stream-1", table_options=table_options
    )
    second = jp.Fluid(
        "CO2", method="bicubic", identifier="stream-2", table_options=table_options
    )
    _, axes = draw(first)
    draw(second, axes=axes)
    assert len(axes._jaxprop_phase_sources) == 2
    assert len(axes.lines) == 4
    sources = list(axes._jaxprop_phase_sources.values())
    assert {source.sat_liq.identifier for source in sources} == {"stream-1", "stream-2"}
    draw(first, axes=axes, plot_saturation_line=False, plot_critical_point=False)
    assert sum(line.get_visible() for line in axes.lines) == 2
    _, other_axes = draw(first)
    assert len(other_axes.lines) == 2 and len(axes.lines) == 4


def test_bicubic_source_plot_rejects_changed_reference_convention(table_options):
    import CoolProp.CoolProp as CP

    fluid = jp.Fluid("CO2", method="bicubic", table_options=table_options)
    try:
        CP.set_reference_state("CO2", "ASHRAE")
        with pytest.raises(jp.InvalidConfigurationError, match="reference convention"):
            draw(fluid)
    finally:
        CP.set_reference_state("CO2", "DEF")


def test_perfect_gas_and_binary_models_reject_own_phase_diagrams():
    gas = jp.Fluid("CO2", method="perfect_gas", model_options={"T_ref": 600.0})
    binary = jp.Fluid(
        "parcels",
        method="binary_mixture",
        model_options={
            "components": (gas, gas),
            "mass_fractions": (0.4, 0.6),
        },
    )
    for fluid in (gas, binary):
        with pytest.raises(
            jp.UnsupportedOperationError, match="saturation envelope"
        ) as captured:
            draw(fluid)
        assert captured.value.model == fluid.model


def test_unavailable_references_and_properties_remain_explicit(source):
    from dataclasses import replace

    with pytest.raises(jp.UnsupportedPropertyError, match="phase-diagram"):
        draw(source, x_prop="unknown-property")
    unavailable = replace(
        source.references.triple_liquid,
        available=False,
        reason="test source has no liquid endpoint",
    )
    object.__setattr__(
        source, "_references", replace(source.references, triple_liquid=unavailable)
    )
    with pytest.raises(jp.UnavailableReferencePointError):
        draw(source, plot_saturation_line=False, plot_triple_point_liquid=True)


def test_pseudo_pure_endpoints_are_not_labeled_as_true_triple_points():
    air = jp.Fluid("Air")
    figure, axes = draw(
        air,
        plot_triple_point_liquid=True,
        plot_triple_point_vapor=True,
        show_in_legend=True,
    )
    elements = air.graphic_elements[axes]
    assert (
        elements["triple_point_liquid"].get_label()
        == "Lower saturation endpoint (liquid)"
    )
    assert (
        elements["triple_point_vapor"].get_label()
        == "Lower saturation endpoint (vapor)"
    )
    figure.canvas.draw()
