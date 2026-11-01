"""FAO-56 reference-equation tests for agri.features.physics.

What this protects against
--------------------------
A silent unit or coefficient change in the irrigation physics. The irrigation
module publishes a recommendation, not an accuracy percentage, so the only
defence against a plausible-but-wrong water balance is the published FAO-56
worked examples plus the closure identity of the daily balance. If the 0.408
radiation term, the 900 wind term or the 0.000665 psychrometric factor drifts,
these tests stop the change before it reaches a field.

Reference values are the FAO-56 (Allen et al., 1998) published examples and are
compared with a one percent relative tolerance because the published numbers are
rounded to three decimals.
"""

from __future__ import annotations

import itertools
from datetime import date

import pytest

from agri.common.errors import DataValidationError
from agri.features import physics

pytestmark = pytest.mark.unit

TOLERANCE = 0.01


def assert_reference(actual: float, expected: float, tolerance: float = TOLERANCE) -> None:
    """Assert a published FAO-56 value, allowing for its printed rounding."""
    assert actual == pytest.approx(expected, rel=tolerance), f"got {actual!r}, expected {expected!r}"


def reference_soil() -> physics.SoilProfile:
    """The medium-textured loam the platform defaults to: TAW 108 mm, RAW 59.4 mm."""
    return physics.SoilProfile(
        field_capacity=0.32,
        wilting_point=0.14,
        root_depth=0.6,
        depletion_fraction=0.55,
    )


# --------------------------------------------------------------- Eq. 8, 11-14
def test_saturation_vapour_pressure_matches_fao56_examples() -> None:
    assert_reference(physics.saturation_vapour_pressure(20.0), 2.338)
    assert_reference(physics.saturation_vapour_pressure(30.0), 4.243)


def test_vapour_pressure_slope_matches_fao56_example() -> None:
    assert_reference(physics.vapour_pressure_slope(20.0), 0.1440)


def test_atmospheric_pressure_matches_fao56_examples() -> None:
    assert_reference(physics.atmospheric_pressure(0.0), 101.3)
    assert_reference(physics.atmospheric_pressure(1000.0), 89.9)


def test_psychrometric_constant_matches_fao56_examples() -> None:
    assert_reference(physics.psychrometric_constant(0.0), 0.0674)
    assert_reference(physics.psychrometric_constant(1000.0), 0.0598)


def test_psychrometric_constant_uses_the_correct_pressure_factor() -> None:
    # gamma = 0.000665 * P, so it must scale exactly with atmospheric pressure.
    ratio = physics.psychrometric_constant(1000.0) / physics.psychrometric_constant(0.0)
    assert ratio == pytest.approx(
        physics.atmospheric_pressure(1000.0) / physics.atmospheric_pressure(0.0), rel=1e-12
    )


# ------------------------------------------------------------- Eq. 21-25, 19
def test_extraterrestrial_radiation_matches_fao56_example() -> None:
    assert_reference(physics.extraterrestrial_radiation(0.0, 81), 37.6)


def test_daylight_hours_matches_fao56_example_at_the_equator() -> None:
    assert_reference(physics.daylight_hours(0.0, 81), 12.0)


def test_clear_sky_radiation_is_a_fraction_of_extraterrestrial_radiation() -> None:
    # Rso = (0.75 + 2e-5 * z) * Ra, so clear-sky radiation is always below Ra.
    ra = physics.extraterrestrial_radiation(17.385, 172)
    rso = physics.clear_sky_radiation(17.385, 172, elevation_m=542.0)
    assert 0.0 < rso < ra
    assert rso / ra == pytest.approx(0.75 + 2e-5 * 542.0)


def test_net_radiation_never_exceeds_incoming_shortwave() -> None:
    rn, rns, rnl = physics.net_radiation(20.0, 34.0, 24.0, 1.2, 17.385, 172)
    assert rns == pytest.approx(0.77 * 20.0)
    assert rnl >= 0.0
    assert rn <= rns


def test_compute_et0_never_returns_a_negative_value() -> None:
    # A vapour deficit cannot be negative, so ET0 must never go below zero even
    # when the supplied vapour pressure exceeds the saturation value.
    weather = physics.DailyWeather(t_max=20.0, t_min=20.0, ea=5.0, solar_radiation=0.0)
    assert physics.compute_et0(weather, 17.385, 172) >= 0.0


def test_compute_et0_from_temperature_estimated_radiation_is_positive() -> None:
    weather = physics.DailyWeather(t_max=34.0, t_min=24.0, humidity_mean=45.0, solar_radiation=None)
    et0 = physics.compute_et0(weather, 17.385, 172)
    assert et0 > 0.0


# ------------------------------------------------------------- Eq. 34, 83, 84
def test_total_and_readily_available_water_reference_values() -> None:
    assert physics.compute_taw(0.32, 0.14, 0.6) == pytest.approx(108.0)
    assert physics.compute_raw(108.0, 0.55) == pytest.approx(59.4)


def test_compute_ks_boundaries_and_linear_interpolation() -> None:
    taw, raw = 108.0, 59.4
    assert physics.compute_ks(raw, raw, taw) == pytest.approx(1.0)
    assert physics.compute_ks(0.0, raw, taw) == pytest.approx(1.0)
    assert physics.compute_ks(taw, raw, taw) == pytest.approx(0.0)
    assert physics.compute_ks(taw + 10.0, raw, taw) == pytest.approx(0.0)

    midpoint = (raw + taw) / 2.0
    expected = (taw - midpoint) / (taw - raw)
    assert physics.compute_ks(midpoint, raw, taw) == pytest.approx(expected)
    assert 0.0 < physics.compute_ks(midpoint, raw, taw) < 1.0


def test_compute_etc_applies_stress_and_kc() -> None:
    assert physics.compute_etc(5.0, 1.0, 1.0) == pytest.approx(5.0)
    assert physics.compute_etc(5.0, 1.0, 0.5) == pytest.approx(2.5)
    assert physics.compute_etc(5.0, 1.0, 0.0) == pytest.approx(0.0)


# ------------------------------------------------------------------- Eq. 82
def test_kc_interpolated_is_continuous_across_the_four_stages() -> None:
    kc_ini, kc_mid, kc_end = 0.30, 1.15, 0.45
    stages = [25, 35, 50, 30]
    total = sum(stages)
    values = [physics.kc_interpolated(kc_ini, kc_mid, kc_end, day, stages)[0] for day in range(total + 5)]

    assert physics.kc_interpolated(kc_ini, kc_mid, kc_end, 0, stages) == (kc_ini, "initial")
    assert values[25] == pytest.approx(kc_ini)
    assert values[60] == pytest.approx(kc_mid)
    assert values[110] == pytest.approx(kc_mid)
    assert values[140] == pytest.approx(kc_end)
    assert physics.kc_interpolated(kc_ini, kc_mid, kc_end, 140, stages)[1] == "post_harvest"

    steepest_slope = max(abs(kc_mid - kc_ini) / stages[1], abs(kc_end - kc_mid) / stages[3])
    largest_daily_step = max(abs(later - earlier) for earlier, later in itertools.pairwise(values))
    assert largest_daily_step <= steepest_slope + 1e-9

    development = values[25:60]
    assert all(later >= earlier for earlier, later in itertools.pairwise(development))


def test_kc_interpolated_rejects_malformed_stage_definitions() -> None:
    with pytest.raises(DataValidationError):
        physics.kc_interpolated(0.3, 1.15, 0.45, 10, [25, 35, 50])
    with pytest.raises(DataValidationError):
        physics.kc_interpolated(0.3, 1.15, 0.45, -1, [25, 35, 50, 30])


# ------------------------------------------------------------------- Eq. 47
def test_wind_speed_at_2m_converts_a_high_measurement_downwards() -> None:
    assert physics.wind_speed_at_2m(3.0, 10.0) < 3.0
    assert physics.wind_speed_at_2m(3.0, 10.0) > 0.0
    assert physics.wind_speed_at_2m(3.0, 2.0) == pytest.approx(3.0)


def test_wind_speed_at_2m_rejects_a_non_positive_height() -> None:
    with pytest.raises(DataValidationError):
        physics.wind_speed_at_2m(3.0, 0.0)


# ------------------------------------------------------------------- Eq. 85
def test_water_balance_closes_over_a_thirty_day_series() -> None:
    # The root zone starts at 40 mm depletion and never refills past field
    # capacity, so no deep percolation occurs and the closure identity must hold
    # to floating-point precision over the whole series.
    soil = physics.SoilProfile(
        field_capacity=0.32,
        wilting_point=0.14,
        root_depth=0.6,
        depletion_fraction=0.55,
        initial_depletion=40.0,
    )
    weather = [
        physics.DailyWeather(
            t_max=28.0 + (day % 5),
            t_min=17.0 + (day % 4),
            humidity_mean=55.0,
            rainfall=6.0 if day % 7 == 0 else 0.0,
            wind_speed_2m=2.0,
        )
        for day in range(30)
    ]
    irrigation = [0.0] * 30
    irrigation[10] = 30.0
    irrigation[20] = 25.0

    result = physics.compute_water_balance(
        weather, soil, 17.385, date(2024, 6, 1), irrigation_depths=irrigation
    )

    assert len(result.daily) == 30
    assert result.total_et0 > 0.0
    assert result.total_irrigation == pytest.approx(55.0)
    assert result.total_rain == pytest.approx(5 * 6.0 * 0.8)
    assert result.total_deep_percolation == 0.0
    assert abs(result.closure_error_mm) < 1e-6
    assert result.final_depletion >= 0.0


def test_water_balance_closure_identity_holds_when_deep_percolation_occurs() -> None:
    # A single 60 mm rain event on a dry profile percolates past field capacity;
    # the percolated depth must be added back in the closure identity, not
    # subtracted, or the reported error would be twice the percolation.
    soil = reference_soil()
    weather = [physics.DailyWeather(t_max=24.0, t_min=14.0, humidity_mean=70.0, rainfall=60.0)]
    result = physics.compute_water_balance(weather, soil, 17.385, date(2024, 6, 1))

    assert result.total_deep_percolation > 0.0
    assert abs(result.closure_error_mm) < 1e-6


def test_water_balance_rejects_a_mismatched_kc_series() -> None:
    soil = reference_soil()
    weather = [physics.DailyWeather(t_max=30.0, t_min=20.0) for _ in range(5)]
    with pytest.raises(DataValidationError):
        physics.compute_water_balance(weather, soil, 17.385, date(2024, 6, 1), kc_series=[1.0, 1.0])


# ---------------------------------------------------------------- Eq. 34-40
def test_recommend_irrigation_within_raw_applies_no_water() -> None:
    soil = reference_soil()
    recommendation = physics.recommend_irrigation(40.0, soil)
    assert recommendation.taw_mm == pytest.approx(108.0)
    assert recommendation.raw_mm == pytest.approx(59.4)
    assert recommendation.should_irrigate is False
    assert recommendation.depth_mm == 0.0
    assert recommendation.urgency == "none"
    assert recommendation.steps


def test_recommend_irrigation_at_the_raw_boundary_applies_no_water() -> None:
    recommendation = physics.recommend_irrigation(59.4, reference_soil())
    assert recommendation.should_irrigate is False
    assert recommendation.depth_mm == 0.0


def test_recommend_irrigation_beyond_raw_irrigates_and_caps_the_depth() -> None:
    soil = reference_soil()

    capped = physics.recommend_irrigation(200.0, soil, max_application_mm=40.0)
    assert capped.should_irrigate is True
    assert capped.depth_mm == pytest.approx(40.0)
    assert capped.depth_mm < 200.0
    assert capped.urgency == "now"

    uncapped = physics.recommend_irrigation(80.0, soil, max_application_mm=100.0)
    assert uncapped.should_irrigate is True
    assert uncapped.depth_mm == pytest.approx(80.0)


def test_recommend_irrigation_credits_forecast_rain_up_to_the_deficit() -> None:
    soil = reference_soil()
    credited = physics.recommend_irrigation(80.0, soil, forecast_rain_mm=50.0)
    assert credited.should_irrigate is False
    assert credited.depth_mm == 0.0


def test_recommend_irrigation_can_refuse_the_forecast_credit() -> None:
    soil = reference_soil()
    no_credit = physics.recommend_irrigation(80.0, soil, forecast_rain_mm=50.0, allow_forecast_credit=False)
    assert no_credit.should_irrigate is True
    assert no_credit.depth_mm == pytest.approx(40.0)
