"""Load model routes and select an inference endpoint for a request."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from random import random
from typing import Any

import yaml
from fastapi import HTTPException

DEFAULT_ROUTES_PATH = Path(__file__).with_name("routes.yaml")


def load_routes(routes_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the model-to-endpoint routing configuration.

    `MODEL_ROUTES_PATH` can point at a different configuration file in a
    deployment environment. Passing `routes_path` is useful for tests.
    """
    config_path = Path(
        routes_path or os.getenv("MODEL_ROUTES_PATH", DEFAULT_ROUTES_PATH)
    )
    try:
        with config_path.open(encoding="utf-8") as route_file:
            config = yaml.safe_load(route_file) or {}
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model route configuration was not found: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"Invalid YAML in model route configuration: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError("Model route configuration must be a YAML mapping")

    models = config.get("models")
    aliases = config.get("aliases", {})
    if not isinstance(models, dict) or not models:
        raise RuntimeError(
            "Model route configuration requires a non-empty 'models' mapping"
        )
    if not isinstance(aliases, dict):
        raise RuntimeError("'aliases' must be a mapping when provided")

    for model_name, route in models.items():
        _validate_route(model_name, route)

    for alias, model_name in aliases.items():
        if not isinstance(model_name, str) or model_name not in models:
            raise RuntimeError(
                f"Alias {alias!r} must reference a model configured in 'models'"
            )

    return {"models": models, "aliases": aliases}


def choose_endpoint(model: str, tenant: str | None = None) -> str:
    """Resolve a requested model and tenant to its Ray Serve base URL."""
    config = load_routes()
    model_name = config["aliases"].get(model, model)
    route = config["models"].get(model_name)  # get route
    if route is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")

    canary = route.get("canary")
    if canary and random() < canary["weight"]:
        return canary["url"]

    tenant_override = route.get("tenants", {}).get(tenant)
    if tenant_override:
        return tenant_override["url"]
    return route["url"]


def _validate_route(model_name: str, route: object) -> None:
    """
    Validates the model route, canary route,"""
    if not isinstance(route, Mapping):
        raise RuntimeError(f"Route for model {model_name!r} must be a mapping")
    _validate_endpoint_url(f"Route for model {model_name!r}", route.get("url"))

    canary = route.get("canary")
    if canary is not None:
        if not isinstance(canary, Mapping):
            raise RuntimeError(
                f"Canary route for model {model_name!r} must be a mapping"
            )
        _validate_endpoint_url(
            f"Canary route for model {model_name!r}", canary.get("url")
        )
        weight = canary.get("weight")
        if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
            raise RuntimeError(
                f"Canary weight for model {model_name!r} must be between 0 and 1"
            )

    tenants = route.get("tenants", {})
    if not isinstance(tenants, Mapping):
        raise RuntimeError(f"Tenant routes for model {model_name!r} must be a mapping")
    for tenant, tenant_route in tenants.items():
        if not isinstance(tenant_route, Mapping):
            raise RuntimeError(
                f"Tenant route {tenant!r} for model {model_name!r} must be a mapping"
            )
        _validate_endpoint_url(
            f"Tenant route {tenant!r} for model {model_name!r}",
            tenant_route.get("url"),
        )


def _validate_endpoint_url(route_name: str, url: object) -> None:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise RuntimeError(f"{route_name} requires an HTTP(S) 'url'")
