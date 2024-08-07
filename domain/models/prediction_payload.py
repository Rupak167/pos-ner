from pydantic import BaseModel


class PredictionPayload(BaseModel):
    sentence: str