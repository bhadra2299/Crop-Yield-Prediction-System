"""Smart Crop AI - agricultural decision-support platform.

Ten modules (nine trained, one pretrained inference-only) plus a FAO-56 irrigation
engine, served through a versioned FastAPI gateway.

The public import surface is intentionally small::

    from agri.common import BasePredictor, ModuleConfig, load_module_config

Subpackages
-----------
common        contract, config, logging, registry, metrics, splits, validation
ingestion     external data pullers (NASA POWER, OpenWeather) with caching
features      shared feature builders - train and serve use the same code path
modules       one package per model, uniform train/evaluate/predict layout
orchestrator  chains modules for /v1/full-recommendation with partial failure
api           FastAPI app, routers, auth, rate limiting
monitoring    drift, data quality, prediction logging
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
