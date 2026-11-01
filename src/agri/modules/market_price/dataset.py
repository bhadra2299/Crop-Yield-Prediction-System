"""Loading and shaping the mandi price series.

Separated from the predictor because the loading rules carry the most important
caveats in the module, and burying them inside _load() would hide them.

The source file has a 20-month hole. It is not a formatting problem that a
forward-fill can paper over: there are no observations at all between 2019-12-31
and 2021-09. Any series built by interpolating across that hole would invent two
years of prices, and every forecast derived from it would be fiction. So the
loader refuses to interpolate, picks the longest continuous segment, and reports
the gap it worked around.

The frame is read once and passed down. Training fits hundreds of commodity/state
series, and re-reading a 59 MB CSV per series turns a thirty-second job into a
half-hour one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from agri.common.errors import DataValidationError
from agri.common.logging import get_logger

_log = get_logger("modules.market_price.dataset")

#: Tolerated gap when deciding whether a record is continuous - covers weekends
#: and mandi holidays, not multi-month outages.
MAX_GAP_DAYS = 3

#: The columns this module needs. Declared once so a schema change fails loudly.
REQUIRED_COLUMNS = ("commodity_name", "state", "market", "modal_price", "date")


@dataclass
class PriceSeries:
    """A cleaned daily price series for one commodity and state."""

    commodity: str
    state: str
    series: pd.Series
    source_rows: int
    markets_used: int
    gap_days: int
    span_start: date
    span_end: date

    def to_dict(self) -> dict[str, object]:
        return {
            "commodity": self.commodity,
            "state": self.state,
            "observations": len(self.series),
            "source_rows": self.source_rows,
            "markets_used": self.markets_used,
            "span_start": self.span_start.isoformat(),
            "span_end": self.span_end.isoformat(),
            "unobserved_span_days": self.gap_days,
        }


def load_market_frame(path: Path) -> pd.DataFrame:
    """Read the market CSV once and normalise it. The only place the file is touched."""
    if not path.is_file():
        raise DataValidationError("Market price dataset is missing", context={"path": str(path)})

    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            "Market price dataset is missing required columns",
            context={"missing": missing, "path": str(path)},
        )

    frame["commodity_name"] = frame["commodity_name"].astype(str).str.strip()
    frame["state"] = frame["state"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["modal_price"] = pd.to_numeric(frame["modal_price"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame[frame["modal_price"] > 0]

    _log.info("market frame loaded", extra={"rows": len(frame)})
    return frame


def series_from_frame(
    frame: pd.DataFrame,
    commodity: str,
    state: str,
    *,
    min_observations: int = 60,
) -> PriceSeries:
    """Build a daily median modal-price series from an already-loaded frame.

    The median across markets is used rather than the mean: a single distressed
    mandi reporting an outlier would drag a mean around, and the median is what a
    farmer would recognise as the going rate.

    No interpolation, no forward-fill, no resampling-with-fill. Missing days stay
    missing; find_continuous_segment decides what is usable.
    """
    subset = frame[(frame["commodity_name"] == commodity) & (frame["state"] == state)]

    if subset.empty:
        available = sorted(frame["commodity_name"].unique().tolist())[:20]
        raise DataValidationError(
            f"No price records for commodity {commodity!r} in state {state!r}",
            context={"commodity": commodity, "state": state, "example_commodities": available},
        )

    source_rows = len(subset)
    markets = int(subset["market"].nunique())
    daily = subset.groupby("date")["modal_price"].median().sort_index()
    span_days = int((daily.index[-1] - daily.index[0]).days + 1)

    if len(daily) < min_observations:
        raise DataValidationError(
            f"Only {len(daily)} distinct price days for {commodity} in {state}; "
            f"at least {min_observations} are needed to fit a trend",
            context={"commodity": commodity, "state": state, "observations": len(daily)},
        )

    return PriceSeries(
        commodity=commodity,
        state=state,
        series=daily,
        source_rows=source_rows,
        markets_used=markets,
        gap_days=span_days - len(daily),
        span_start=daily.index[0].date(),
        span_end=daily.index[-1].date(),
    )


def ranking_from_frame(frame: pd.DataFrame, max_series: int) -> list[tuple[str, str, int]]:
    """The commodity/state pairs with the most observed days, most first."""
    counts = frame.groupby(["commodity_name", "state"])["date"].nunique().sort_values(ascending=False)
    return [(commodity, state, int(count)) for (commodity, state), count in list(counts.items())[:max_series]]


def load_series(
    path: Path,
    commodity: str,
    state: str,
    *,
    min_observations: int = 60,
) -> PriceSeries:
    """Convenience wrapper: read the file and build one series.

    Use series_from_frame when fitting many series so the file is read once.
    """
    frame = load_market_frame(path)
    return series_from_frame(frame, commodity, state, min_observations=min_observations)
