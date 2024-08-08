# Documentation for Converting a Keras Model to ONNX Format
#### Converting a trained Keras model to the ONNX format using the tf2onnx library, enabling the model to be deployed across various platforms that support ONNX.

## Script Explanation
### 1. Importing Required Libraries
```python
import tensorflow as tf
import tf2onnx
```

**tensorflow:** Used to load the Keras model.

**tf2onnx:** Used to convert the loaded Keras model to ONNX format.

### 2. Loading the Keras Model
```python
keras_model_path = 'model/ner_pos_model.keras'
model = tf.keras.models.load_model(keras_model_path)
```
**keras_model_path:** The path to the saved Keras model. Ensure this path points to the model you want to convert.

**tf.keras.models.load_model:** Loads the model from the specified path into a TensorFlow/Keras model object.
### 3. Specifying the ONNX Conversion Parameters
```python
onnx_model_path = 'model/ner_pos_model.onnx'
spec = (tf.TensorSpec((None, 50), tf.float32, name="input"),)
```
**onnx_model_path:** The path where the converted ONNX model will be saved.

**spec:** Defines the input signature for the model. It specifies that the model expects input tensors with shape (None, 50) where None allows for variable batch sizes, and 50 corresponds to the sequence length used during model training. The input data type is tf.float32.
### 4. Converting the Keras Model to ONNX Format
```python
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
```
**tf2onnx.convert.from_keras:** Converts the Keras model to ONNX format.

**model:** The loaded Keras model.

**input_signature:** The input signature defined earlier.

**opset:** Specifies the ONNX opset version. 13 is commonly used and compatible with most platforms.

### 5. Saving the ONNX Model
```python
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

**onnx_model_path:** The path where the ONNX model will be saved.
SerializeToString(): Serializes the model to a binary string format, which is then written to the specified file.

### 6. Confirming the Conversion
```python
print(f"Model saved to {onnx_model_path}")
```
This statement prints the path to the saved ONNX model, confirming that the conversion process was successful.