import os
from contextlib import asynccontextmanager

import joblib
import keras
import keras.layers
import anthropic
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import state
from app.config import MODEL_PATH, REG_MODEL_PATH, ANTHROPIC_API_KEY, STATIC_DIR
from app.routes import predict as predict_router
from app.routes import explain as explain_router

# ---------------------------------------------------------------------------
# Keras 3 compatibility patch — must run before load_model is called.
# Model was saved with a Keras 3 version that added 'quantization_config' to
# Dense layers; newer Keras removed it. Patch from_config to drop it silently.
# ---------------------------------------------------------------------------
_orig_dense_from_config = keras.layers.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig_dense_from_config(cls, config)

keras.layers.Dense.from_config = _patched_dense_from_config


# ---------------------------------------------------------------------------
# Application lifespan — load models once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state.model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    if os.path.exists(REG_MODEL_PATH):
        state.reg_model = joblib.load(REG_MODEL_PATH)
    if ANTHROPIC_API_KEY:
        state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Sleep Staging API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(predict_router.router)
app.include_router(explain_router.router)


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