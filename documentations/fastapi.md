# Documentation for POS and NER Prediction API
**application.py** has the Fast API that performs Part-of-Speech (POS) tagging and Named Entity Recognition (NER) on input sentences. It uses a pre-trained model to generate predictions.

## Overview
### The API is designed to:
- **Accept POST requests containing sentences to be processed for POS and NER predictions.**
- **Return the POS and NER tags for the input sentences.**

Prerequisites
**Environs:** For environment variable management.

**Logging:** To log application events.

**Custom Modules:** PredictionPayload, NER_POS_Prediction, and get_logger are custom modules assumed to be part of the project.


## Code Explanation
### 1. Importing Necessary Modules
```python
from environs import Env
from fastapi import FastAPI, status

from domain.models import PredictionPayload
from domain.pos_ner_prediction import NER_POS_Prediction

from utils import get_logger
```

**environs:** Manages environment variables.

**fastapi:** FastAPI framework for creating the web API.

**Custom imports:** These are project-specific modules for logging, model loading, and handling prediction payloads and result entity.

### 2. Setting Up Environment Variables
```python
env = Env()
env.read_env()
is_debugging = env.bool("DEBUG", False)
```
**Env():** Initializes the environment handler.

**read_env():** Reads environment variables from a .env file.

**is_debugging:** A boolean to toggle debugging mode, based on the DEBUG environment variable.

### 3. Configuring Logging and FastAPI Application

```python
logger = get_logger(__name__)
app = FastAPI(debug=is_debugging, title="POS and NER API")
logger.info("Application initialized")
```

**get_logger:** Custom function to initialize logging.

**FastAPI():** Initializes the FastAPI application with debugging mode and a title.

**logger.info():** Logs that the application has been initialized.

### 4. POST Endpoint for Predictions
```python 
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

```
**@app.post("/predict"):** Defines a POST endpoint at /predict.

**PredictionPayload:** A Pydantic model for validating incoming JSON payloads.

**NER_POS_Prediction:** Custom class for loading the model and making predictions.

**payload.sentence:** Extracts the sentence from the incoming payload.

**model.predict():** Makes the call to domain layer and generates predictions for the input sentence.


**Return Value:** A list of prediction results, serialized using the model_dump method.

### 5. GET Endpoint for Health Check

```python 
@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}
```

**@app.get("/healthcheck"):** Defines a GET endpoint at /healthcheck.

**status.HTTP_200_OK:** Ensures that a successful request returns HTTP 200.

**Return Value:** A JSON response indicating the API's health status.


## Custom Entity
These are the custom models to store payload, result and tokens. Located in **domain/models**

### 1. Token

```python
from pydantic import BaseModel

class Token(BaseModel):
    Token: str
    POS_Tag: str
    NER_Tag: str
```

### 2. Result

```python
from pydantic import BaseModel, Field
from .token import Token

class Result(BaseModel):
    predictions : list[Token]
```

### 3. Prediction Payload
```python
from pydantic import BaseModel

class PredictionPayload(BaseModel):
    sentence: str
```

# Inference Code
This class is implemented to predict the pos and ner tags which is same as **test.py** explained in [test.md](test.md) 

**Located in domain/pos_ner_prediction**
