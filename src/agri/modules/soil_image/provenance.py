"""Single source of truth for the data-honesty statements this module must repeat.

The corpus under data/raw/soil_images is PROCEDURALLY GENERATED. Three separate
artefacts have to say so, and they must not drift apart:

  1. the registry manifest (what produced this model),
  2. the metrics report (what the headline score actually measures),
  3. every prediction response (what the caller is entitled to assume).

Keeping the wording here means a reviewer can grep one file for the caveat rather
than trust three copies of it.
"""

from __future__ import annotations

#: Hard flag: every manifest, report and response carries this as a boolean.
SYNTHETIC_DATA = True

#: Where the images came from, quoted from data/raw/soil_images/README.md.
DATA_PROVENANCE = (
    "Procedurally generated synthetic textures produced by "
    "data/raw/soil_images/generate_soil_images.py (fractal Perlin noise, Voronoi "
    "granularity, simulated desiccation cracks, root fibers and pebble inclusions). "
    "These are not photographs of real soil in real fields."
)

#: The limitation a reader must apply to any number this module reports.
LIMITATION = (
    "Trained and evaluated only on procedural synthetic textures. The reported "
    "macro_f1 measures how separable those synthetic textures are; it is NOT a "
    "field-accuracy claim and must not be quoted as one. Any accuracy statement for "
    "real soil photographs requires re-labelled field data and re-measurement."
)

#: Short version attached to each serving response.
SERVING_CAVEAT = (
    "This model was trained on synthetic soil textures, not on field photographs. "
    "Treat the predicted class as a hint for a real photograph, not as a measured result."
)


def limitation_bundle() -> dict[str, object]:
    """The provenance block written into manifests and metrics reports."""
    return {
        "synthetic_data": SYNTHETIC_DATA,
        "data_provenance": DATA_PROVENANCE,
        "limitation": LIMITATION,
    }
