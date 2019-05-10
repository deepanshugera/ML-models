import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import pickle
from tensorflow.keras.utils import to_categorical

def unpickle_test(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

t = unpickle_test('../dataset/Imagenet32_val/val_data')
test = t['data']
test_x = test.reshape(test.shape[0], 3, 32, 32)
test_x = test_x.transpose(0, 2, 3, 1)
test_x = test_x.astype(float)/255.0

test_y = t['labels']
test_y = np.reshape(test_y, (len(test_y), 1))
test_y = to_categorical(test_y, num_classes=1001)

interpreter = tf.lite.Interpreter(model_path=str("quantized_model.tflite"))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def eval_model(interpreter):
  total_seen = 0
  num_correct = 0

  for i in range(50000):
    total_seen += 1
    img = tf.stack([test_x[i]])
    img = tf.cast(img, tf.uint8)
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    #print ((predictions[0].argmax() + 1) , " : " , np.argmax((test_y[i]) + 1))
    if ((predictions[0].argmax() + 1)  == (np.argmax(test_y[i]) + 1)):
      num_correct += 1

    if total_seen % 500 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)
print(eval_model(interpreter))
