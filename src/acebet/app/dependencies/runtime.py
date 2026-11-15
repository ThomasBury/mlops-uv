"""Runtime asset dependencies shared by API routers."""

from typing import Annotated

from fastapi import Depends, Request

from acebet.app.runtime import RuntimeAssets


def get_runtime_assets(request: Request) -> RuntimeAssets:
    """Return the runtime asset container owned by the current application."""
    return request.app.state.runtime_assets


RuntimeAssetsDep = Annotated[RuntimeAssets, Depends(get_runtime_assets)]
