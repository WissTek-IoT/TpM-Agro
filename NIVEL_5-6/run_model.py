import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('NIVEL_4/model_data.keras')

# # Show the model architecture
# model.summary()

# Functions
import os
validation_data_file_location   = os.path.join(os.path.dirname(__file__), '../NIVEL_4/validation_data.txt')
def read_validation_data():
    X = []
    Y = []

    # Open validation file and stores its inputs and respective outputs
    validation_data_file = open(validation_data_file_location, 'r', encoding='utf-8-sig')
    for line in validation_data_file:
        # Separates data from semicolon
        line = line.strip()
        data_line = line.split(';')

        # Separates input (X) and output (Y) data
        X.append(data_line[:14])
        Y.append(data_line[-2:-1])

    # Converts data into a numpy array
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(int).flatten()
    
    return X, Y

X_val, Y_val    = read_validation_data()
probability_model               = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Gives an example for a prediction based on a single data
current_input = X_val[55]
current_input = (np.expand_dims(current_input, 0))

current_prediction  = probability_model.predict(current_input)
prediction_label    = np.argmax(current_prediction)

# Extracts the predicted value from the predicted label
predicted_pump_signal  = prediction_label & 0x01 
predicted_light_signal = prediction_label >> 1

print(("\n\nPredicted output: {}\n"+
       "Real output: {}\n"+
       "Pump Signal: {} | Light Signal: {}\n" +
       "Level of Confidence: {}%").format(prediction_label,
                                           Y_val[55], 
                                           predicted_pump_signal,
                                           predicted_light_signal,
                                           100*current_prediction[0][prediction_label]))
