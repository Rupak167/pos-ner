# Named Entity Recognition (NER) and Parts of Speech (POS) tagging.

## Install Dependencies to run locally on Ubuntu

## 1. Install poetry: 

```shell
pip install poetry
```

## 2. Clone the repository:

```shell
git clone https://github.com/Rupak167/pos-ner.git
```
## 3. Open the project in a code editor and run:
```shell
poetry install
```
## 4. Select the poetry environment:
```shell
poetry shell
```

# Now for training and testing run the scripts.

## Run using make
```shell
make train
```
Test the model using
```shell
make test
```

## or Run using 

```shell
python training.py
```
Test the model using
```shell
python test.py
```
***Note: Change the input text first.***

# Prepare ONNX model and inference
## Prepare ONNX model

```shell
make prepare-onnx
```
***or***
```shell
python prepare_onnx_model.py
```
## Inference using ONNX model
```shell
make infer-onnx
```
***or***
```shell
python inference_onnx_model.py
```

# Run the inference API developed using FastAPI

## 1. Run the API
```shell
uvicorn application:app --workers 2 --host 0.0.0.0 --port 8000
```
## 2. Open a API testing app like Postman and set

## Endpoint

```shell
http://0.0.0.0:8000/predict
```
## Payload
```shell
{
    "sentence": "আপনার ইনপুট বাক্য এখানে"
}
```


# To check api health

## 1. Make a GET request
```shell
http://0.0.0.0:8000/healthcheck
```

# To run using docker

## 1. Build and Run using Make

```shell
make run
```

**Or run**
```shell
docker compose up --build
```

***Use the same process to make api request***


## Documentation Overview
[Training Process:](documentations/data_preparation_and_training.md) Detailed guide on how to prepare data and train the POS and NER model.

[Testing the Model:](documentations/test.md) Instructions for testing the model and evaluating its performance.

[Converting to ONNX Format:](documentations/prepare_onnx_model.md)
 Steps to convert the trained model to ONNX format for optimized deployment.

[Testing the ONNX Model:](documentations/ONNX_testing.md) Guidelines for testing the model after converting it to ONNX.

[FastAPI Implementation:](documentations/fastapi.md) Documentation on the FastAPI interface for the model inference.