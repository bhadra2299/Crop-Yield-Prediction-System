"""FAO-56 irrigation scheduling.

This module implements the FAO-56 dual-step water balance directly from the
published equations rather than wrapping a third-party package, for two reasons:
the reference equations are the thing we are audited against, and the standard
worked examples in FAO-56 Chapter 4 can be reproduced as unit tests, which a
wrapper would hide behind its own abstraction.

References
----------
FAO Irrigation and Drainage Paper 56, "Crop evapotranspiration - Guidelines for
computing crop water requirements", Allen et al., 1998.

    Eq. 6    FAO-56 Penman-Monteith reference evapotranspiration (ET0)
    Eq. 8    mean saturation vapour pressure from Tmin/Tmax
    Eq. 11   slope of the saturation vapour pressure curve
    Eq. 12   atmospheric pressure from elevation
    Eq. 14   psychrometric constant
    Eq. 19   net radiation
    Eq. 20   net shortwave radiation from solar radiation
    Eq. 21   net longwave radiation (Brunt-style cloudiness correction)
    Eq. 23   clear-sky solar radiation
    Eq. 24   extraterrestrial radiation
    Eq. 25   daylight hours
    Eq. 34-40 soil water balance and total available water
    Eq. 82   Kc interpolation between growth stages

Every function is pure and unit-explicit. FAO-56 is specified in millimetres per
day, degrees Celsius, metres per second and megajoules per square metre per day,
and mixing units is the single most common way a water balance silently goes
wrong. Where a conversion is needed it is named, not inlined.

No accuracy percentage is claimed anywhere in this module. There is no public
ground truth for "the correct irrigation depth" on a given field, so the module
is validated for physical consistency against the FAO-56 worked examples, not
against field observations. See docs/MODULES/irrigation.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from agri.common.errors import DataValidationError

# Stefan-Boltzmann constant, MJ K^-4 m^-2 day^-1
STEFAN_BOLTZMANN = 4.903e-9
#: Solar constant, MJ m^-2 min^-1
SOLAR_CONSTANT = 0.0820
#: Latent heat of vaporisation at 20 C, MJ kg^-1 (FAO-56 Eq. 4 uses 2.45)
LATENT_HEAT = 2.45
#: Albedo for the hypothetical reference crop (FAO-56 Eq. 38)
ALBEDO = 0.23
#: Von Karman constant is folded into the 900/0.34 coefficients of Eq. 6


@dataclass(frozen=True)
class DailyWeather:
    """One day of weather. Units are FAO-56 units; the parser is responsible for
    converting provider payloads, not this class."""

    t_max: float
    t_min: float
    #: Relative humidity, percent
    humidity_mean: float | None = None
    #: Actual vapour pressure, kPa. Derived from humidity when absent.
    ea: float | None = None
    #: Wind speed at 2 m height, m/s
    wind_speed_2m: float = 2.0
    #: Incoming solar radiation, MJ m^-2 day^-1. Estimated from temperature range when absent.
    solar_radiation: float | None = None
    #: Rainfall, mm
    rainfall: float = 0.0
    #: Station elevation, m (used for atmospheric pressure)
    elevation: float = 0.0

    @property
    def t_mean(self) -> float:
        return (self.t_max + self.t_min) / 2.0


@dataclass
class SoilProfile:
    """Soil water holding properties for the root zone."""

    #: Volumetric water content at field capacity, m^3/m^3
    field_capacity: float
    #: Volumetric water content at wilting point, m^3/m^3
    wilting_point: float
    #: Effective rooting depth, m
    root_depth: float
    #: Depletion fraction p for the crop (FAO-56 Table 22), 0-1
    depletion_fraction: float = 0.55
    #: Readily available water already depleted at the start, mm
    initial_depletion: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.wilting_point < self.field_capacity <= 1.0:
            raise DataValidationError(
                "Soil water content must satisfy wilting_point < field_capacity within 0..1",
                context={"wilting_point": self.wilting_point, "field_capacity": self.field_capacity},
            )
        if self.root_depth <= 0:
            raise DataValidationError("root_depth must be positive", context={"root_depth": self.root_depth})


@dataclass
class WaterBalanceResult:
    """Outcome of a step-by-step soil water balance."""

    daily: list[dict[str, float]] = field(default_factory=list)
    total_et0: float = 0.0
    total_etc: float = 0.0
    total_rain: float = 0.0
    total_irrigation: float = 0.0
    total_deep_percolation: float = 0.0
    final_depletion: float = 0.0
    #: Reported error against the closure identity, mm. Must be ~0.
    closure_error_mm: float = 0.0
    stress_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_et0": round(self.total_et0, 3),
            "total_etc": round(self.total_etc, 3),
            "total_rain": round(self.total_rain, 3),
            "total_irrigation": round(self.total_irrigation, 3),
            "total_deep_percolation": round(self.total_deep_percolation, 3),
            "final_depletion": round(self.final_depletion, 3),
            "closure_error_mm": round(self.closure_error_mm, 9),
            "stress_days": self.stress_days,
            "days": len(self.daily),
        }


@dataclass
class IrrigationRecommendation:
    """The actionable answer, with the arithmetic that produced it."""

    should_irrigate: bool
    depth_mm: float
    reason: str
    urgency: str  # none | monitor | soon | now
    depletion_mm: float
    raw_mm: float
    taw_mm: float
    forecast_rain_mm: float
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_irrigate": self.should_irrigate,
            "depth_mm": round(self.depth_mm, 2),
            "reason": self.reason,
            "urgency": self.urgency,
            "depletion_mm": round(self.depletion_mm, 2),
            "raw_mm": round(self.raw_mm, 2),
            "taw_mm": round(self.taw_mm, 2),
            "forecast_rain_mm": round(self.forecast_rain_mm, 2),
            "steps": self.steps,
        }


# --------------------------------------------------------------- Eq. 8, 11-14
def saturation_vapour_pressure(t: float) -> float:
    """e0(T) in kPa, FAO-56 Eq. 8."""
    return 0.6108 * math.exp((17.27 * t) / (t + 237.3))


def mean_saturation_vapour_pressure(t_max: float, t_min: float) -> float:
    """es in kPa, the mean of e0(Tmax) and e0(Tmin) - FAO-56 Eq. 12 note."""
    return (saturation_vapour_pressure(t_max) + saturation_vapour_pressure(t_min)) / 2.0


def vapour_pressure_slope(t_mean: float) -> float:
    """Delta in kPa per degree C, FAO-56 Eq. 13."""
    numerator = 4098.0 * (0.6108 * math.exp((17.27 * t_mean) / (t_mean + 237.3)))
    return numerator / ((t_mean + 237.3) ** 2)


def atmospheric_pressure(elevation_m: float) -> float:
    """P in kPa, FAO-56 Eq. 7."""
    return 101.3 * (((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26)


def psychrometric_constant(elevation_m: float) -> float:
    """Gamma in kPa per degree C, FAO-56 Eq. 8."""
    return 0.000665 * atmospheric_pressure(elevation_m)


def actual_vapour_pressure(t_max: float, t_min: float, humidity_mean: float | None) -> float:
    """ea in kPa.

    Preferred estimator is ea from the mean relative humidity (FAO-56 Eq. 17);
    when humidity is missing the dewpoint-free approximation ea = e0(Tmin) from
    Eq. 48 is used, which FAO-56 recommends for arid sites and is conservative
    for humid ones.
    """
    if humidity_mean is None:
        return saturation_vapour_pressure(t_min)
    humidity = min(max(humidity_mean, 0.0), 100.0)
    return (
        saturation_vapour_pressure(t_min) * (humidity / 100.0)
        if False
        else ((saturation_vapour_pressure(t_max) + saturation_vapour_pressure(t_min)) / 2.0)
        * (humidity / 100.0)
    )


# ------------------------------------------------------------- Eq. 21-25, 19
def extraterrestrial_radiation(latitude_deg: float, day_of_year: int) -> float:
    """Ra in MJ m^-2 day^-1, FAO-56 Eq. 21."""
    phi = math.radians(latitude_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
    # Sunset hour angle, Eq. 25
    argument = max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta)))
    omega = math.acos(argument)
    return (
        (24.0 * 60.0 / math.pi)
        * SOLAR_CONSTANT
        * dr
        * (omega * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(omega))
    )


def daylight_hours(latitude_deg: float, day_of_year: int) -> float:
    """N in hours, FAO-56 Eq. 24."""
    phi = math.radians(latitude_deg)
    delta = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
    argument = max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta)))
    return (24.0 / math.pi) * math.acos(argument)


def clear_sky_radiation(latitude_deg: float, day_of_year: int, elevation_m: float = 0.0) -> float:
    """Rso in MJ m^-2 day^-1, FAO-56 Eq. 22."""
    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    return (0.75 + 2.0e-5 * elevation_m) * ra


def solar_radiation_from_temperature_range(
    latitude_deg: float, day_of_year: int, t_max: float, t_min: float
) -> float:
    """Rs estimated from the temperature range, FAO-56 Eq. 35.

    Used only when a provider does not supply measured solar radiation, which is
    the common case for OpenWeather's free tier. The station is assumed coastal
    (kRs = 0.19); an inland site should supply measured radiation instead.
    """
    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    return 0.19 * math.sqrt(max(t_max - t_min, 0.0)) * ra


def net_radiation(
    solar_radiation: float,
    t_max: float,
    t_min: float,
    ea: float,
    latitude_deg: float,
    day_of_year: int,
    elevation_m: float = 0.0,
) -> tuple[float, float, float]:
    """Return (Rn, Rns, Rnl) in MJ m^-2 day^-1, FAO-56 Eq. 19/20/21.

    The cloudiness factor is clamped at 1.0. Values above it appear when measured
    Rs exceeds the theoretical clear-sky value, which happens at high altitude
    and with sensor error; letting it through makes Rnl negative and produces a
    reference ET larger than the incoming energy, which is unphysical.
    """
    rns = (1.0 - ALBEDO) * solar_radiation
    t_max_k = t_max + 273.16
    t_min_k = t_min + 273.16
    rso = clear_sky_radiation(latitude_deg, day_of_year, elevation_m)
    cloudiness = min(max(solar_radiation / rso, 0.0), 1.0) if rso > 0 else 1.0
    rnl = (
        STEFAN_BOLTZMANN
        * ((t_max_k**4 + t_min_k**4) / 2.0)
        * (0.34 - 0.14 * math.sqrt(max(ea, 0.0)))
        * (1.35 * cloudiness - 0.35)
    )
    return rns - rnl, rns, rnl


# ------------------------------------------------------------------- Eq. 6
def compute_et0(
    weather: DailyWeather,
    latitude_deg: float,
    day_of_year: int,
) -> float:
    """Reference crop evapotranspiration ET0 in mm/day, FAO-56 Eq. 6.

    ET0 = (0.408 * Delta * (Rn - G) + gamma * 900/(T+273) * u2 * (es - ea))
          / (Delta + gamma * (1 + 0.34 * u2))

    Soil heat flux G is taken as 0 for the daily step, which FAO-56 Eq. 42
    explicitly permits.
    """
    t_mean = weather.t_mean
    delta = vapour_pressure_slope(t_mean)
    gamma = psychrometric_constant(weather.elevation)
    es = mean_saturation_vapour_pressure(weather.t_max, weather.t_min)
    ea = (
        weather.ea
        if weather.ea is not None
        else actual_vapour_pressure(weather.t_max, weather.t_min, weather.humidity_mean)
    )
    # Vapour deficit cannot be negative: supersaturated air is a measurement
    # artefact, and a negative deficit would subtract energy from the crop.
    vapour_deficit = max(es - ea, 0.0)

    solar = weather.solar_radiation
    if solar is None or solar <= 0:
        solar = solar_radiation_from_temperature_range(
            latitude_deg, day_of_year, weather.t_max, weather.t_min
        )
    rn, _, _ = net_radiation(
        solar, weather.t_max, weather.t_min, ea, latitude_deg, day_of_year, weather.elevation
    )

    wind = max(weather.wind_speed_2m, 0.0)
    numerator = 0.408 * delta * rn + gamma * (900.0 / (t_mean + 273.0)) * wind * vapour_deficit
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    et0 = numerator / denominator
    # A reference ET below zero has no physical meaning for a daily step.
    return max(et0, 0.0)


def wind_speed_at_2m(wind_speed: float, measurement_height_m: float) -> float:
    """Convert wind measured at another height to 2 m, FAO-56 Eq. 47."""
    if measurement_height_m <= 0:
        raise DataValidationError("measurement_height_m must be positive")
    if abs(measurement_height_m - 2.0) < 1e-9:
        return wind_speed
    return wind_speed * (4.87 / math.log(67.8 * measurement_height_m - 5.42))


# ------------------------------------------------------------------- Eq. 82
def kc_interpolated(
    kc_ini: float,
    kc_mid: float,
    kc_end: float,
    stage_day: int,
    stage_lengths: list[int],
) -> tuple[float, str]:
    """Crop coefficient for a day in the season, and the stage it falls in.

    FAO-56 Eq. 82 linear interpolation across the four growth stages
    (initial, development, mid-season, late-season).
    """
    if len(stage_lengths) != 4:
        raise DataValidationError("stage_lengths must have four entries", context={"got": len(stage_lengths)})
    initial, development, mid, late = stage_lengths
    if stage_day < 0:
        raise DataValidationError("stage_day cannot be negative")

    if stage_day < initial:
        return kc_ini, "initial"
    if stage_day < initial + development:
        # Linear ramp from Kc_ini to Kc_mid over the development stage.
        fraction = (stage_day - initial) / development if development else 1.0
        return kc_ini + fraction * (kc_mid - kc_ini), "development"
    if stage_day < initial + development + mid:
        return kc_mid, "mid_season"
    if stage_day < initial + development + mid + late:
        fraction = (stage_day - initial - development - mid) / late if late else 1.0
        return kc_mid + fraction * (kc_end - kc_mid), "late_season"
    return kc_end, "post_harvest"


def compute_etc(et0: float, kc: float, ks: float = 1.0) -> float:
    """Crop evapotranspiration, FAO-56 Eq. 56/81: ETc = Ks * Kc * ET0."""
    water_stress = min(max(ks, 0.0), 1.0)
    return max(water_stress * kc * et0, 0.0)


# ----------------------------------------------------------------- Eq. 34-40
def compute_taw(field_capacity: float, wilting_point: float, root_depth_m: float) -> float:
    """Total available water in the root zone, mm, FAO-56 Eq. 82.

    TAW = 1000 * (theta_fc - theta_wp) * Zr, where the 1000 converts metres of
    water to millimetres.
    """
    return 1000.0 * (field_capacity - wilting_point) * root_depth_m


def compute_raw(taw_mm: float, depletion_fraction: float) -> float:
    """Readily available water, mm, FAO-56 Eq. 83: RAW = p * TAW."""
    return max(0.0, min(depletion_fraction, 1.0)) * taw_mm


def compute_ks(depletion_mm: float, raw_mm: float, taw_mm: float) -> float:
    """Water stress coefficient, FAO-56 Eq. 84.

    Ks = 1 while the root zone is within RAW, then falls linearly to 0 at TAW.
    """
    if taw_mm <= raw_mm:
        return 1.0 if depletion_mm <= raw_mm else 0.0
    if depletion_mm <= raw_mm:
        return 1.0
    if depletion_mm >= taw_mm:
        return 0.0
    return (taw_mm - depletion_mm) / (taw_mm - raw_mm)


def compute_water_balance(
    weather_series: list[DailyWeather],
    soil: SoilProfile,
    latitude_deg: float,
    start_date: date,
    kc_series: list[float] | None = None,
    irrigation_depths: list[float] | None = None,
    effective_rain_fraction: float = 0.8,
) -> WaterBalanceResult:
    """Run the FAO-56 daily root-zone water balance (Eq. 85).

    Dr,i = Dr,i-1 - (P - RO) - I - CR + ETc + DP

    Runoff is taken as zero for a level field, which FAO-56 permits. Capillary
    rise is taken as zero, which is the documented assumption for a deep water
    table being absent. Deep percolation is whatever exceeds field capacity.
    """
    taw = compute_taw(soil.field_capacity, soil.wilting_point, soil.root_depth)
    raw = compute_raw(taw, soil.depletion_fraction)
    depletion = max(0.0, soil.initial_depletion)

    result = WaterBalanceResult()
    result.final_depletion = depletion

    if len(weather_series) == 0:
        return result

    if kc_series is not None and len(kc_series) != len(weather_series):
        raise DataValidationError(
            "kc_series must match the weather series length",
            context={"kc": len(kc_series), "weather": len(weather_series)},
        )

    for index, weather in enumerate(weather_series):
        current_date = start_date.fromordinal(start_date.toordinal() + index)
        day_of_year = current_date.timetuple().tm_yday
        et0 = compute_et0(weather, latitude_deg, day_of_year)
        kc = kc_series[index] if kc_series is not None else 1.0
        ks = compute_ks(depletion, raw, taw)
        etc = compute_etc(et0, kc, ks)

        # Only the effective fraction of rain refills the root zone; the rest
        # runs off or is intercepted.
        effective_rain = max(weather.rainfall, 0.0) * effective_rain_fraction
        irrigation = irrigation_depths[index] if irrigation_depths is not None else 0.0

        depletion = depletion - effective_rain - irrigation + etc
        deep_percolation = 0.0
        if depletion < 0:
            # Water beyond field capacity drains below the root zone.
            deep_percolation = -depletion
            depletion = 0.0

        stressed = ks < 0.999
        result.daily.append(
            {
                "day": index + 1,
                "et0": round(et0, 3),
                "kc": round(kc, 3),
                "ks": round(ks, 3),
                "etc": round(etc, 3),
                "effective_rain": round(effective_rain, 3),
                "irrigation": round(irrigation, 3),
                "deep_percolation": round(deep_percolation, 3),
                "depletion": round(depletion, 3),
                "stressed": 1.0 if stressed else 0.0,
            }
        )
        result.total_et0 += et0
        result.total_etc += etc
        result.total_rain += effective_rain
        result.total_irrigation += irrigation
        result.total_deep_percolation += deep_percolation
        result.stress_days += 1 if stressed else 0

    result.final_depletion = depletion
    initial = max(0.0, soil.initial_depletion)
    # Closure identity, following FAO-56 Eq. 85:
    #   Dr,i = Dr,i-1 - (P - RO) - I + ETc + DPi
    # Deep percolation enters with a PLUS sign: in the recursion above it is the
    # amount added back to lift a negative depletion to zero, so it increases Dr.
    # Subtracting it here double-counted it and made the error equal to twice the
    # percolated depth.
    expected = (
        initial
        + result.total_etc
        - result.total_rain
        - result.total_irrigation
        + result.total_deep_percolation
    )
    result.closure_error_mm = depletion - expected
    return result


def recommend_irrigation(
    depletion_mm: float,
    soil: SoilProfile,
    forecast_rain_mm: float = 0.0,
    max_application_mm: float = 40.0,
    effective_rain_fraction: float = 0.8,
    allow_forecast_credit: bool = True,
) -> IrrigationRecommendation:
    """Decide whether and how much to irrigate, with the arithmetic exposed.

    The decision is a threshold on root-zone depletion, not a learned model:
    irrigate when depletion exceeds RAW, applying enough to refill to field
    capacity, capped by the application limit so a single pass cannot waterlog
    the profile.

    Forecast rain is credited only when it is expected before the crop would
    suffer - and the credit is capped at the deficit, because rain cannot reduce
    a deficit below zero.
    """
    taw = compute_taw(soil.field_capacity, soil.wilting_point, soil.root_depth)
    raw = compute_raw(taw, soil.depletion_fraction)
    depletion = max(0.0, depletion_mm)

    credited_rain = 0.0
    if allow_forecast_credit and forecast_rain_mm > 0:
        credited_rain = min(forecast_rain_mm * effective_rain_fraction, depletion)

    effective_depletion = depletion - credited_rain

    steps = [
        f"TAW = 1000 x ({soil.field_capacity} - {soil.wilting_point}) x {soil.root_depth} m = {taw:.1f} mm",
        f"RAW = p x TAW = {soil.depletion_fraction} x {taw:.1f} = {raw:.1f} mm",
        f"current depletion = {depletion:.1f} mm",
    ]
    if credited_rain > 0:
        steps.append(
            f"forecast rain credit = min({forecast_rain_mm} x {effective_rain_fraction}, {depletion:.1f}) = {credited_rain:.1f} mm"
        )

    if effective_depletion <= raw:
        reason = (
            f"Root-zone depletion ({effective_depletion:.1f} mm) is within the readily "
            f"available water ({raw:.1f} mm); the crop is not water stressed."
        )
        if effective_depletion > raw * 0.8:
            urgency = "monitor"
            reason += " It is close to the threshold, so re-check after the next forecast."
        else:
            urgency = "none"
        return IrrigationRecommendation(
            should_irrigate=False,
            depth_mm=0.0,
            reason=reason,
            urgency=urgency,
            depletion_mm=depletion,
            raw_mm=raw,
            taw_mm=taw,
            forecast_rain_mm=forecast_rain_mm,
            steps=steps,
        )

    required = effective_depletion
    depth = min(required, max_application_mm)
    fraction_of_raw = effective_depletion / raw if raw else float("inf")
    urgency = "now" if fraction_of_raw >= 1.35 else "soon"

    steps.append(f"depletion exceeds RAW by {effective_depletion - raw:.1f} mm")
    steps.append(
        f"net depth required = {required:.1f} mm, capped at {max_application_mm:.1f} mm -> {depth:.1f} mm"
    )
    if depth < required:
        steps.append(
            f"a single pass cannot refill the profile; about {math.ceil(required / max_application_mm)} passes "
            "are needed, spaced to avoid ponding"
        )

    return IrrigationRecommendation(
        should_irrigate=True,
        depth_mm=depth,
        reason=(
            f"Root-zone depletion ({effective_depletion:.1f} mm) exceeds the readily available "
            f"water ({raw:.1f} mm), so the crop is drawing on reserves it cannot fully replace."
        ),
        urgency=urgency,
        depletion_mm=depletion,
        raw_mm=raw,
        taw_mm=taw,
        forecast_rain_mm=forecast_rain_mm,
        steps=steps,
    )


def depletion_from_moisture(volumetric_moisture: float, soil: SoilProfile) -> float:
    """Convert a sensor moisture reading to root-zone depletion, mm.

    Depletion from field capacity is (theta_fc - theta) x Zr x 1000. Readings
    below wilting point are clamped: a sensor in dry, cracking clay can report
    values the crop cannot actually access.
    """
    if not 0.0 <= volumetric_moisture <= 1.0:
        raise DataValidationError(
            "volumetric moisture must be a fraction in 0..1",
            context={"value": volumetric_moisture},
        )
    usable = min(max(volumetric_moisture, soil.wilting_point), soil.field_capacity)
    return max(1000.0 * (soil.field_capacity - usable) * soil.root_depth, 0.0)


def trend_correlation(observed: list[float], modelled: list[float]) -> float:
    """Pearson correlation between two series, for satellite cross-checks.

    Used to sanity-check the modelled depletion trend against an independent
    observation such as NASA SMAP surface soil moisture. A correlation is a
    consistency check, never a validation of irrigation accuracy.
    """
    if len(observed) != len(modelled) or len(observed) < 3:
        return float("nan")
    a = np.asarray(observed, dtype="float64")
    b = np.asarray(modelled, dtype="float64")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
