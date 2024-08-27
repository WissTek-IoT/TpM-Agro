import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Files
abstraction_data_file_location  = os.path.join(os.path.dirname(__file__), '../NIVEL_4/abstraction_data.txt')
validation_data_file_location   = os.path.join(os.path.dirname(__file__), '../NIVEL_4/validation_data.txt')
model_file_location             = os.path.join(os.path.dirname(__file__), '..NIVEL_4/')

# Functions
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

def read_abstraction_data():
    X = []
    Y = []

    # Open abstraction file and stores its inputs and respective outputs
    abstraction_data_file = open(abstraction_data_file_location, 'r', encoding='utf-8-sig')
    for line in abstraction_data_file:
        # Separates data from semicolon
        line = line.strip()
        data_line = line.split(';')

        # Separates input (X) and output (Y) data
        X.append(data_line[:14])
        Y.append(data_line[-2:-1])

    # Converts data into a numpy array
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(int).flatten()

    print(( "Input Shape: {} \n"+
            "Output Shape: {}").format(X.shape, Y.shape))
    
    return X, Y

# Code
# Stores input variables into X and respective outputs into Y
X, Y            = read_abstraction_data()
X_val, Y_val    = read_validation_data()

# Normalize input values X
normalization = tf.keras.layers.Normalization(axis=-1)  # Creates a normalization instance
normalization.adapt(X)                                  # Learns mean and variance from input dataset X

# Creates the Machine Learning Model
model = tf.keras.Sequential([
    normalization,
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X, Y, epochs=10)

# # Prints the cost over epoch
# # summarize history for loss
# # print(history.history['accuracy'])
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'loss'], loc='upper left')
# plt.show()

# Validates model using the given training set
test_loss, test_accuracy = model.evaluate(X_val,  Y_val, verbose=2)
print('\nTest accuracy:', test_accuracy)

# Computes predicted output for each validation set's input
probability_model               = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions_for_validation_set  = probability_model.predict(X_val)

# Gives an example for a prediction based on a single data
single_input = X_val[55]
single_input = (np.expand_dims(single_input,0))

single_prediction = probability_model.predict(single_input)
single_prediction = np.argmax(single_prediction)
print(("Predicted output: {}\n"+
       "Real output: {}").format(single_prediction, Y_val[55]))

model.save('NIVEL_4/model_data.keras')
print("O modelo foi salvo na pasta NIVEL_4")