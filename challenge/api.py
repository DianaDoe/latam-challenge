import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List
from pydantic.types import Enum

from challenge.model import DelayModel

app = fastapi.FastAPI()


class FlightTypeEnum(str, Enum):
    N = "N"
    I = "I"


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: FlightTypeEnum
    MES: int = Field(..., title="Mes", ge=1, le=12)


class FlightPayload(BaseModel):
    flights: List


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(payload: FlightPayload) -> dict:
    model = DelayModel()
    data = []
    for flight in payload.flights:
        try:
             Flight(**flight)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        data.append(flight)

    data = pd.DataFrame(data)
    features = model.preprocess(
        data=data
    )

    predicted_targets = model.predict(
        features=features
    )

    return {"predict": predicted_targets}
