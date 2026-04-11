import logging
import os
from contextlib import asynccontextmanager

import joblib
import keras
import keras.layers
import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import state
from app.config import MODEL_PATH, REG_MODEL_PATH, ANTHROPIC_API_KEY, GEMINI_API_KEY, STATIC_DIR
from app.routes import predict as predict_router
from app.routes import explain as explain_router
from app.logging_config import configure_logging

configure_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keras 3 compatibility patches — must run before load_model is called.
#
# Patch 1: Model was saved with Keras 3.0-3.5 which stored the Functional
# class at keras.src.engine.functional. Keras 3.6+ moved it to
# keras.src.models.functional. Register the old path as an alias so that
# checkpoints from older Keras versions load without errors.
# ---------------------------------------------------------------------------
import sys
import types
import keras.src.models.functional as _keras_functional_module

_engine_stub = types.ModuleType("keras.src.engine")
_engine_stub.functional = _keras_functional_module
sys.modules.setdefault("keras.src.engine", _engine_stub)
sys.modules.setdefault("keras.src.engine.functional", _keras_functional_module)

# ---------------------------------------------------------------------------
# Patch 2: Model was saved with a Keras 3 version that added
# 'quantization_config' to Dense layers; newer Keras removed it.
# Drop it silently during deserialization.
# ---------------------------------------------------------------------------
_orig_dense_from_config = keras.layers.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig_dense_from_config(cls, config)

keras.layers.Dense.from_config = _patched_dense_from_config

# ---------------------------------------------------------------------------
# Patch 3: BatchNormalization 'axis' was serialized as a list [2] in older
# Keras 3.x; newer Keras expects a scalar integer. Unwrap single-element
# lists before calling the original from_config.
# ---------------------------------------------------------------------------
_orig_bn_from_config = keras.layers.BatchNormalization.from_config.__func__

@classmethod
def _patched_bn_from_config(cls, config):
    axis = config.get("axis")
    if isinstance(axis, list) and len(axis) == 1:
        config["axis"] = axis[0]
    return _orig_bn_from_config(cls, config)

keras.layers.BatchNormalization.from_config = _patched_bn_from_config

# ---------------------------------------------------------------------------
# Patch 4: LSTM 'time_major' argument was removed in Keras 3.6+.
# Drop it silently.
# ---------------------------------------------------------------------------
_orig_lstm_from_config = keras.layers.LSTM.from_config.__func__

@classmethod
def _patched_lstm_from_config(cls, config):
    config.pop("time_major", None)
    config.pop("implementation", None)
    return _orig_lstm_from_config(cls, config)

keras.layers.LSTM.from_config = _patched_lstm_from_config


# ---------------------------------------------------------------------------
# Application lifespan — load models once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading CNN+LSTM model", extra={"path": MODEL_PATH})
    state.model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    log.info("CNN+LSTM model loaded")

    if os.path.exists(REG_MODEL_PATH):
        state.reg_model = joblib.load(REG_MODEL_PATH)
        log.info("Regression model loaded", extra={"path": REG_MODEL_PATH})
    else:
        log.warning("Regression model not found — /predict/quick-score will be unavailable",
                    extra={"path": REG_MODEL_PATH})

    if ANTHROPIC_API_KEY:
        state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        log.info("Anthropic client initialised — AI explanations enabled")
    else:
        log.warning("ANTHROPIC_API_KEY not set — skipping Anthropic client")

    if GEMINI_API_KEY:
        from google import genai as google_genai
        state.gemini_model = google_genai.Client(api_key=GEMINI_API_KEY)
        log.info("Gemini client initialised — AI explanations enabled")
    else:
        log.warning("GEMINI_API_KEY not set — /explain will use rule-based fallback")

    log.info("Startup complete")
    yield
    log.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Sleep Staging API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(predict_router.router)
app.include_router(explain_router.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info("Request", extra={"method": request.method, "path": request.url.path})
    response = await call_next(request)
    log.info("Response", extra={"method": request.method, "path": request.url.path, "status": response.status_code})
    return response


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cnn_lstm_loaded":      state.model is not None,
        "regression_loaded":    state.reg_model is not None,
        "ai_explain_available": state.anthropic_client is not None,
    }