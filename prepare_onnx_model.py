import tensorflow as tf
import tf2onnx

keras_model_path = 'model/ner_pos_model.keras'
model = tf.keras.models.load_model(keras_model_path)

onnx_model_path = 'model/ner_pos_model.onnx'
spec = (tf.TensorSpec((None, 50), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model saved to {onnx_model_path}")
