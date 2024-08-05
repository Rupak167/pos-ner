from base64 import b64decode
from io import BytesIO

from environs import Env
from fastapi import FastAPI, status

from utils import get_logger

env = Env()
env.read_env()
is_debugging = env.bool("DEBUG", False)


logger = get_logger(__name__)
app = FastAPI(debug=is_debugging, title="POS and NER API")
logger.info("Application initialized")


@app.post("/predict")
async def get_prediction(payload: dict):
    pass


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}