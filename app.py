import math
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

FEATURES = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
SUPPORTED_VERSIONS = ("v0.1", "v0.2")
MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


def create_app(model_version: str | None = None) -> FastAPI:
    version = model_version if model_version is not None else os.getenv("MODEL_VERSION", "v0.2")
    if version not in SUPPORTED_VERSIONS:
        supported = ", ".join(SUPPORTED_VERSIONS)
        raise ValueError(f"Unsupported MODEL_VERSION {version!r}; choose one of: {supported}")

    model_path = MODEL_DIRECTORY / f"model_{version}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")

    model = joblib.load(model_path)
    application = FastAPI()
    application.state.model = model
    application.state.model_version = version

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model_version": application.state.model_version}

    @application.post("/predict")
    async def predict(payload: PatientFeatures) -> JSONResponse:
        values = [getattr(payload, name) for name in FEATURES]
        features = pd.DataFrame([values], columns=FEATURES)
        prediction = float(application.state.model.predict(features)[0])
        if not math.isfinite(prediction):
            raise RuntimeError("The configured model returned a non-finite prediction.")
        return JSONResponse(content={"prediction": prediction})

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "80")))
