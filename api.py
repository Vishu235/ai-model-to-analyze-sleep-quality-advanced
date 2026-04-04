# Entry point — kept thin so Railway's `uvicorn api:app` command works unchanged.
from app.main import app  # noqa: F401

# Re-export helper functions so existing tests (which do `import api as api_mod`)
# can still call api_mod._bandpass_filter, api_mod._validate_signal, etc.
from app.services.signal import (
    _bandpass_filter,
    _validate_signal,
    _extract_spectral_features,
)
from app.services.scoring import (
    _compute_metrics,
    _rule_based_explain,
)