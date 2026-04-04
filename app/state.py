# Global model handles — populated during FastAPI lifespan startup.
# Importing this module before startup is safe; all values will be None.
from typing import Optional, Any

model: Optional[Any]           = None   # CNN+LSTM Keras model
reg_model: Optional[Any]       = None   # RandomForest regression model
anthropic_client: Optional[Any] = None  # anthropic.Anthropic client