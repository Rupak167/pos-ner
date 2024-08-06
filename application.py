from base64 import b64decode
from io import BytesIO

from environs import Env
from fastapi import FastAPI, status

from domain.models import PredictionPayload
from domain.pos_ner_prediction import NER_POS_Prediction

from utils import get_logger

env = Env()
env.read_env()
is_debugging = env.bool("DEBUG", False)

logger = get_logger(__name__)
app = FastAPI(debug=is_debugging, title="POS and NER API")
logger.info("Application initialized")


@app.post("/predict")
async def get_prediction(payload: PredictionPayload):
    try:
        model = NER_POS_Prediction(
            model_path=env.str("MODEL_PATH"),
            data_path=env.str("DATA_PATH"),
            max_len=env.int("MAX_LEN")
        )
        sentence = payload.sentence
        results = model.predict(sentence)
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}
    else:
        return [result.model_dump() for result in results]


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}