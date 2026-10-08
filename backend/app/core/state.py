"""
app/core/state.py
-----------------
Tiny process-global runtime state, separate from `config.Settings` (which holds
static configuration). Kept in its own module so both `main.py` (writer) and the
API layer (readers) can import it without creating an import cycle through the
heavy `detector` module.

`model_ready` reflects whether the open-vocabulary detection model preloaded
successfully. The lifespan startup hook sets it (R-8); `/readyz` (O-3) and the
open-vocabulary detect path read it to fast-fail instead of retrying an
expensive model load on every request.
"""

from __future__ import annotations

# Set True once the open-vocabulary detector has loaded; False if preload failed
# (e.g. missing weights / GPU OOM). Trained-model detection does NOT depend on
# this flag — it loads its own weights on demand.
model_ready: bool = False
