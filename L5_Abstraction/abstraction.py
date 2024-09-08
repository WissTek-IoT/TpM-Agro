# IMPORTS
import os
# Edit tensorflow flags before importing it
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import random
import tensorflow   as tf
import numpy        as np
from enum       import Enum
from datetime   import datetime

# FILES
application_data_file_location      = os.path.join(os.path.dirname(__file__), '../L4_Storage/application_data.txt')
abstraction_data_file_location      = os.path.join(os.path.dirname(__file__), '../L4_Storage/abstraction_data.txt')
training_data_file_location         = os.path.join(os.path.dirname(__file__), '../L4_Storage/training_data.txt'   )
validation_data_file_location       = os.path.join(os.path.dirname(__file__), '../L4_Storage/validation_data.txt' )
testing_data_file_location          = os.path.join(os.path.dirname(__file__), '../L4_Storage/testing_data.txt'    )
commands_file_location              = os.path.join(os.path.dirname(__file__), '../L4_Storage/commands.txt'        )
prediction_queue_file_location      = os.path.join(os.path.dirname(__file__), '../L4_Storage/prediction_queue.txt')
pump_waiting_keras_model_file_location     = "L4_Storage/pump_waiting_model.keras"
pump_activating_keras_model_file_location  = "L4_Storage/pump_activating_model.keras"
light_keras_model_file_location            = "L4_Storage/light_model.keras"
pump_waiting_tflite_model_file_location    = "L4_Storage/pump_waiting_model.tflite"
pump_activating_tflite_model_file_location = "L4_Storage/pump_activating_model.tflite"
light_tflite_model_file_location           = "L4_Storage/light_model.tflite"

# CLASSES
class data_indexes(Enum):
    DATE_INDEX              = 0
    HOUR_INDEX              = 1

    TEMPERATURE_INDEX       = 2
    HUMIDITY_INDEX          = 3

    VISIBLE_LIGHT_INDEX     = 4
    IR_LIGHT_INDEX          = 5
    UV_INDEX                = 6

    CONTROL_MODE_INDEX      = 7

    PUMP_ENABLED_INDEX      = 8
    PUMP_WAITING_INDEX      = 9
    PUMP_ACTIVATING_INDEX   = 10
    LIGHT_ENABLED_INDEX     = 11
date_index              = data_indexes.DATE_INDEX.value
hour_index              = data_indexes.HOUR_INDEX.value
temperature_index       = data_indexes.TEMPERATURE_INDEX.value
humidity_index          = data_indexes.HUMIDITY_INDEX.value
visible_light_index     = data_indexes.VISIBLE_LIGHT_INDEX.value
ir_light_index          = data_indexes.IR_LIGHT_INDEX.value
uv_index_index          = data_indexes.UV_INDEX.value
control_mode_index      = data_indexes.CONTROL_MODE_INDEX.value
pump_enabled_index      = data_indexes.PUMP_ENABLED_INDEX.value
pump_waiting_index      = data_indexes.PUMP_WAITING_INDEX.value
pump_activating_index   = data_indexes.PUMP_ACTIVATING_INDEX.value
light_enabled_index     = data_indexes.LIGHT_ENABLED_INDEX.value

# THRESHOLD FOR OUTLIERS
temperature_outlier     = 200.0
humidity_outlier        = 200.0
visible_light_outlier   = 20000
ir_light_outlier        = 20000
uv_index_outlier        = 100.0

# DATETIME FORMAT
time_format = "%d-%m-%Y;%H:%M:%S"

# SCALE OUTPUTS FOR REGRESSION MODEL
pump_waiting_attenuation = 100.0
pump_activating_attenuation = 10.0

# FUNCTIONS
def get_seconds(time_str):
    """Gets seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def store_data_into_file(data, file_location):
    """Stores given data into .txt file"""
    file = open(file_location, 'w+')
    if (file.writable()):
        for i in range(len(data)):
            # Converts all data to string and then separate it by semicolons
            temp        = list(map(str, data[i]))
            temp        .append("\n")
            data_line   = ";".join(temp)

            # Stores data into text file
            file.write(data_line)
    file.close()

def set_seed(seed):
    # Set fixed seeds for model consistency
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def read_application_data_from_file(file_location, is_prediction_queue=False):
    data        = []

    data_file   = open(file_location, 'r', encoding='utf-8-sig')
    for line in data_file:
        # Separates data from semicolon
        line = line.strip()
        data_line = line.split(';')

        # The first value of prediction queue is not a data line, so we need to treat it differently
        if (is_prediction_queue and len(data_line) == 1):
            data.append(int(line))
        else:
            # Stores each data into its respective list
            date            = data_line[date_index]
            hour            = data_line[hour_index]
            temperature     = float   (data_line[temperature_index])
            humidity        = float   (data_line[humidity_index])
            visible_light   = int     (data_line[visible_light_index])
            ir_light        = int     (data_line[ir_light_index])
            uv_index        = float   (data_line[uv_index_index])
            control_mode    = int     (data_line[control_mode_index])
            pump_enabled    = int     (data_line[pump_enabled_index])
            pump_waiting    = int     (data_line[pump_waiting_index])
            pump_activating = int     (data_line[pump_activating_index])
            light_enabled   = int     (data_line[light_enabled_index])

            # If none of the values is an outlier, store the data line
            if (
                temperature     < temperature_outlier   and
                humidity        < humidity_outlier      and
                visible_light   < visible_light_outlier and
                ir_light        < ir_light_outlier      and
                uv_index        < uv_index_outlier      
            ):
                data.append([
                    date,
                    hour,
                    temperature,
                    humidity,
                    visible_light,
                    ir_light,
                    uv_index,
                    control_mode,
                    pump_enabled,
                    pump_waiting,
                    pump_activating,
                    light_enabled
                ])
    data_file.close()

    # Separates data counter from actual data
    if (is_prediction_queue):
        if(len(data)<4): return -1, -1
        data_counter = data[0]
        data = np.array(data[1:])
        return data, data_counter
    
    data = np.array(data)
    return data

def read_abstraction_data_from_file(file_location):
    abstraction        = []

    abstraction_file   = open(file_location, 'r', encoding='utf-8-sig')
    for line in abstraction_file:
        # Separates data from semicolon
        line = line.strip()
        abstraction_line = line.split(';')

        hour             = float(abstraction_line[0])
        temperature      = float(abstraction_line[1])
        humidity         = float(abstraction_line[2])
        visible_light    = float(abstraction_line[3])
        ir_light         = float(abstraction_line[4])
        uv_index         = float(abstraction_line[5])
        elapsed_time     = float(abstraction_line[6])
        time_interval    = float(abstraction_line[7])
        d_temperature    = float(abstraction_line[8])
        d_humidity       = float(abstraction_line[9])
        d_visible_light  = float(abstraction_line[10])
        d_ir_light       = float(abstraction_line[11])
        d_uv_index       = float(abstraction_line[12])
        pump_enabled     = float(abstraction_line[13])
        pump_waiting     = float(abstraction_line[14])
        pump_activating  = float(abstraction_line[15])
        light_enabled    = float(abstraction_line[16])

        abstraction.append([
            hour,
            temperature,
            humidity,
            visible_light,
            ir_light,
            uv_index,
            elapsed_time,
            time_interval,
            d_temperature,
            d_humidity,
            d_visible_light,
            d_ir_light,
            d_uv_index,
            pump_enabled,
            pump_waiting,
            pump_activating,
            light_enabled
        ])
    abstraction_file.close()
    abstraction = np.array(abstraction)

    return abstraction

def generate_abstraction_data(application_data, is_prediction_queue=False):
    abstraction_data = []

    # Application Data
    date            = application_data[:, date_index            ] 
    hour            = application_data[:, hour_index            ] 
    temperature     = application_data[:, temperature_index     ].astype(float) 
    humidity        = application_data[:, humidity_index        ].astype(float) 
    visible_light   = application_data[:, visible_light_index   ].astype(int) 
    ir_light        = application_data[:, ir_light_index        ].astype(int)  
    uv_index        = application_data[:, uv_index_index        ].astype(float) 
    control_mode    = application_data[:, control_mode_index    ].astype(int)  
    pump_enabled    = application_data[:, pump_enabled_index    ].astype(int)  
    pump_waiting    = application_data[:, pump_waiting_index    ].astype(int)  
    pump_activating = application_data[:, pump_activating_index ].astype(int)  
    light_enabled   = application_data[:, light_enabled_index   ].astype(int)  

    # Abstraction Data
    hour_in_seconds     = []
    elapsed_time        = []
    time_interval       = []
    d_temperature       = []
    d_humidity          = []
    d_visible_light     = []
    d_ir_light          = []
    d_uv_index          = []

    for i in range(len(application_data)):
        hour_in_seconds.append(get_seconds(hour[i]))

        if (i != 0):
            # Calculate elapsed time
            initial_time    = datetime.strptime(str(date[0] + ';' + hour[0])    ,time_format)   # Merges separated date and hour presented in time_format format into a datetime instance
            current_time    = datetime.strptime(str(date[i] + ';' + hour[i])    ,time_format)   # Same as above
            elapsed_time.append((current_time-initial_time).total_seconds())

            # Calculate time interval
            previous_time   = datetime.strptime(str(date[i-1] + ';' + hour[i-1]), time_format)  # Same as above
            delta_seconds   = (current_time - previous_time).total_seconds()
            time_interval.append(int(delta_seconds))

            # Calculate derivatives
            d_temperature       .append((temperature[i]     - temperature[i-1])     /   time_interval[i])
            d_humidity          .append((humidity[i]        - humidity[i-1])        /   time_interval[i])
            d_visible_light     .append((visible_light[i]   - visible_light[i-1])   /   time_interval[i])
            d_ir_light          .append((ir_light[i]        - ir_light[i-1])        /   time_interval[i])
            d_uv_index          .append((uv_index[i]        - uv_index[i-1])        /   time_interval[i])
        else:
            # We cannot calculate derivatives for the 1st data line
            elapsed_time        .append(0)
            time_interval       .append(9999999)
            d_temperature       .append(0)
            d_humidity          .append(0)
            d_visible_light     .append(0)
            d_ir_light          .append(0)
            d_uv_index          .append(0)
        
        abstraction_data.append([
            hour_in_seconds[i],
            temperature[i],
            humidity[i],
            visible_light[i],
            ir_light[i],
            uv_index[i],
            elapsed_time[i],
            time_interval[i],
            d_temperature[i],
            d_humidity[i],
            d_visible_light[i],
            d_ir_light[i],
            d_uv_index[i],
            pump_enabled[i],
            pump_waiting[i]/pump_waiting_attenuation,
            pump_activating[i]/pump_activating_attenuation,
            light_enabled[i]
        ])
    abstraction_data = np.array(abstraction_data).astype(float)

    if (is_prediction_queue): 
        return abstraction_data[2]
    return abstraction_data

def train_regression_model(model, 
                           training_input, 
                           training_output, 
                           validation_input,
                           validation_output,
                           test_input,
                           test_output,
                           number_of_epochs):
    print("Performing trainning...")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae', 'mse', 'mean_absolute_percentage_error']
    )
    model.fit(
        training_input,
        training_output,
        validation_data=[validation_input, validation_output],
        epochs=number_of_epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)]
    )
    print("Finished training. Model summary:")
    print(model.summary())

    # Print Root Mean Squared Error for testing set
    print("\nEvaluating the model using testing set.")
    model.evaluate(test_input, test_output)

    prediction = model.predict(test_input)
    print("Prediction for first value on set: ", prediction[0])
    print("Prediction for last value on set: ", prediction[-1])

    return model, prediction

def train_classification_model(model, 
                               training_input, 
                               training_output, 
                               validation_input,
                               validation_output,
                               test_input,
                               test_output,
                               number_of_epochs):
    print("Performing trainning...")
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.fit(training_input, 
              training_output, 
              validation_data=[validation_input, validation_output],
              epochs=number_of_epochs)
    print("Finished training.")
    print(model.summary())

    print("\nModel metrics:")
    valid_loss, valid_accuracy = model.evaluate(test_input,  test_output, verbose=0)

    # Computes predicted output for each validation set's input
    probability_model            = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions_for_testing_set  = probability_model.predict(test_input)

    # Prints Accuracy and Loss
    print(f"Accuracy: {round(100*valid_accuracy,3)}%")
    print(f"Loss: {round(100*valid_loss,3)}%")

    return probability_model, predictions_for_testing_set

def evaluate_model_precision_on_activating(output_type, validation_output, predicted_output):
    """Gives metrics on the precision of the given model to activate given actuator."""
    print(f"\nTesting for {output_type} activation precision.")
    times_active_in_validation_set = 0
    times_active_in_prediction_set = 0

    active_in_validation = []
    active_in_prediction = []

    # Calculates how many times the actuator was active on validation set
    for i in range(len(validation_output)):
        if validation_output[i] == 1:
            times_active_in_validation_set += 1
            active_in_validation.append(i)

    # Calculates how many times the actuator was active on prediction set
    for i in range(len(predicted_output)):
        output = np.argmax(predicted_output[i])
        if (output == 1):
            times_active_in_prediction_set += 1
            active_in_prediction.append(i)
    
    # Calculates the number of activations at same time
    matching_activations = 0
    for i in active_in_prediction:
        if (np.argmax(predicted_output[i]) == validation_output[i]):
            matching_activations += 1

    # Prints activation metrics
    print(f"Number of times {output_type} was active on validation set:", times_active_in_validation_set)
    print(f"Number of predicted {output_type} activations using validation set as input for model:", times_active_in_prediction_set)
    print(f"Relative error in {output_type} activation: {round(100*(times_active_in_prediction_set - times_active_in_validation_set)/times_active_in_validation_set, 2)}%")
    print(f"Time difference: {round((times_active_in_prediction_set - times_active_in_validation_set)/60, 2)} minutes.")
    print(f"Matching {output_type} activations: {matching_activations}")

def train_pump_waiting_model(training_data, validation_data, testing_data):
    print("Training pump model.")
    # Reads model variables
        # Reads training data
    X_pump_train = training_data    [:, 0:13]
    Y_pump_train = training_data    [:, 14]
        # Reads validation data
    X_pump_valid = validation_data  [:, 0:13]
    Y_pump_valid = validation_data  [:, 14]
        # Reads testing_data
    X_pump_test  = testing_data     [:, 0:13]
    Y_pump_test  = testing_data     [:, 14]

    # Creates a normalization layer
    normalization = tf.keras.layers.Normalization(axis=-1) 
    normalization.adapt(X_pump_train)

    # Creates the machine learning model
    model = tf.keras.Sequential([
        normalization,
        tf.keras.layers.Dense(27, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(6,  activation='relu'),
        tf.keras.layers.Dense(5,  activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Train model
    pump_model, predictions = train_regression_model(
        model,
        X_pump_train,
        Y_pump_train,
        X_pump_valid,
        Y_pump_valid,
        X_pump_test,
        Y_pump_test,
        number_of_epochs=6
    )

    # Saves model
    pump_model.save(pump_waiting_keras_model_file_location)
        # Converts into a tflite model and saves it as well
    converter            = tf.lite.TFLiteConverter.from_keras_model(pump_waiting_model)
    pump_w_tflite_model  = converter.convert()
    with open(pump_waiting_tflite_model_file_location, 'wb') as f:
        f.write(pump_w_tflite_model)
    print("\nModel was saved on L4_Storage as pump_waiting_model.keras")

def train_pump_activating_model(training_data, validation_data, testing_data):
    print("Training pump model.")
    # Reads model variables
        # Reads training data
    X_pump_train = training_data    [:, 0:13]
    Y_pump_train = training_data    [:, 15]
        # Reads validation data
    X_pump_valid = validation_data  [:, 0:13]
    Y_pump_valid = validation_data  [:, 15]
        # Reads testing_data
    X_pump_test  = testing_data     [:, 0:13]
    Y_pump_test  = testing_data     [:, 15]

    # Creates a normalization layer
    normalization = tf.keras.layers.Normalization(axis=-1) 
    normalization.adapt(X_pump_train)

    # Creates the machine learning model
    model = tf.keras.Sequential([
        normalization,
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(8,  activation='relu'),
        tf.keras.layers.Dense(5,  activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Train model
    pump_activating_model, predictions = train_regression_model(
        model,
        X_pump_train,
        Y_pump_train,
        X_pump_valid,
        Y_pump_valid,
        X_pump_test,
        Y_pump_test,
        number_of_epochs=5
    )

    # Saves model
    pump_activating_model.save(pump_activating_keras_model_file_location)
        # Converts into a tflite model and saves it as well
    converter            = tf.lite.TFLiteConverter.from_keras_model(pump_activating_model)
    pump_a_tflite_model  = converter.convert()
    with open(pump_activating_tflite_model_file_location, 'wb') as f:
        f.write(pump_a_tflite_model)

    print("\nModel was saved on L4_Storage as pump_activating_model.keras")

def train_light_model(training_data, validation_data, testing_data):
    print("Training light model.")
    # Reads data
        # Reads training data
    X_light_train = training_data  [:, [0, 6]]
    Y_light_train = training_data  [:, 16]
        # Reads validation data
    X_light_valid = validation_data[:, [0, 6]]
    Y_light_valid = validation_data[:, 16]
        # Reads testing data
    X_light_test  = testing_data   [:, [0, 6]]
    Y_light_test  = testing_data   [:, 16]

    # Creates a normalization layer
    normalization = tf.keras.layers.Normalization(axis=-1) 
    normalization.adapt(X_light_train)
    
    # Creates the machine learning model
    model = tf.keras.Sequential([
        normalization,
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    # Train model
    light_model, predictions = train_classification_model(
        model, 
        X_light_train,
        Y_light_train,
        X_light_valid,
        Y_light_valid,
        X_light_test,
        Y_light_test,
        number_of_epochs=4
    )
    
    evaluate_model_precision_on_activating("light", Y_light_test, predictions)

    # Save the model.
    model.save(light_keras_model_file_location)
        # Converts into a tflite model and saves it as well
    converter           = tf.lite.TFLiteConverter.from_keras_model(light_model)
    light_tflite_model  = converter.convert()
    with open(light_tflite_model_file_location, 'wb') as f:
        f.write(light_tflite_model)

    print("\nModel was saved on L4_Storage as light_model.keras")

def read_line_from_commands_file(line):
    """Reads commands file and extracts given line.\n
    Returns given line value"""
    commands = []

    commands_file = open(commands_file_location, 'r')
    # Stores all commands into an array
    for line in commands_file:
        # Separates data from colon
        line = line.strip()
        data_line = line.split(';')
        data_line = data_line[1].strip()
        commands.append(data_line)

    commands_file.close()
    if (len(commands) >= 0):
        return float(commands[line])
    
    return 0

def predict_system_output(pump_waiting_model, 
                          input_for_pump_waiting_prediction, 
                          pump_activating_model,
                          input_for_pump_activating_prediction,
                          light_model,
                          input_for_light_prediction
                          ):
    pump_waiting_input              = np.expand_dims(input_for_pump_waiting_prediction, 0)
    pump_activating_input           = np.expand_dims(input_for_pump_activating_prediction, 0)
    light_input                     = np.expand_dims(input_for_light_prediction, 0)

    pump_waiting_prediction         = pump_waiting_model.predict(pump_waiting_input)
    pump_activating_prediction      = pump_activating_model.predict(pump_activating_input)
    light_prediction                = light_model.predict(light_input)

    pump_waiting_output_signal      = round(pump_waiting_attenuation*pump_waiting_prediction[0][0])
    pump_activating_output_signal   = round(pump_activating_attenuation*pump_activating_prediction[0][0])
    light_output_signal             = np.argmax(light_prediction)

    # Extracts the predicted value from the predicted label
    light_confidence_level          = round(light_prediction[0][light_output_signal]*100, 2)

    return (pump_waiting_output_signal, 
            pump_activating_output_signal,
            light_output_signal, 
            light_confidence_level)

def send_predicted_signals(pump_waiting_time, 
                           pump_activating_time, 
                           light, 
                           light_confidence,
                           pump_waiting_confidence=0,
                           pump_activating_confidence=0,
                           ):
    commands_file = open(commands_file_location, 'r')
    commands = commands_file.readlines()
    commands_file.close()

    value_index = max(commands[1].find("0"), commands[1].find("1"))
    commands[2] = commands[2][:value_index] + str(light)                + '\n'
    commands[3] = commands[3][:value_index] + str(pump_waiting_time)    + '\n'
    commands[4] = commands[4][:value_index] + str(pump_activating_time) + '\n'
    commands[8] = commands[8][:value_index] + str(light_confidence)     + '\n'
    # commands[8] = commands[8][:value_index] + str(pump_waiting_confidence)      + '\n'
    # commands[9] = commands[9][:value_index] + str(pump_activating_confidence)   + '\n'

    commands_file = open(commands_file_location, 'w+')
    commands_file.writelines(commands)
    commands_file.close()

# MAIN
running_mode = int(input("Select between: Train ML Model (0) or Run ML Model (1)\n"))
if (running_mode == 0):
    print("\nYou chose to train the Machine Learning Model.")

    # Set seed for stability purposes
    set_seed(42)

    # Processes application data and generate abstraction data
    load_abstraction = input("Do you want to load abstraction from file?[Y/N]\n")
    if (load_abstraction == "Y" or load_abstraction == "y"):
        print("Loading data from abstraction file...")
        abstraction_data = read_abstraction_data_from_file(abstraction_data_file_location)
    else:
        print("Processing abstraction data...")
        application_data = read_application_data_from_file(application_data_file_location)
        abstraction_data = generate_abstraction_data(application_data)
        store_data_into_file(abstraction_data, abstraction_data_file_location)

    # Processes data length and splits abstraction data into training, validation and testing data
    split1 = round(0.7*len(abstraction_data))
    split2 = round((len(abstraction_data) - split1)/2) + split1

    # Generates training and validation data
    training_data   = abstraction_data[:split1]         # Extracts abstraction data from first value to split1
    validation_data = abstraction_data[split1:split2]   # Extracts abstraction data from split1 to split2
    testing_data    = abstraction_data[split2:]         # Extracts abstraction data from split2 to last value
        # And stores them into files
    store_data_into_file(training_data  , training_data_file_location  )
    store_data_into_file(validation_data, validation_data_file_location)
    store_data_into_file(testing_data   , testing_data_file_location   )

    # Prints out information on abstraction data
    print("Abstracion data processed.\n")
    print(f"Total data: {len(abstraction_data)} lines.")
    print(f"70% of total data ({len(training_data)} lines) will be used to train the model.")
    print(f"15% of total data ({len(validation_data)} lines) will be used to validate the model during training.")
    print(f"15% of total data ({len(testing_data)} lines) will be used to test the model accuracy.\n")

    model_to_train = int(input( "Select which model to train    \n" + 
                                "Pump Waiting Interval\t\t(0)\n"+
                                "Pump Activation Interval\t(1)\n"+
                                "Light Activation\t\t(2)\n"))
    if (model_to_train == 0):
        print("You selected to train the pump waiting model.")
        train_pump_waiting_model(training_data, validation_data, testing_data)
    elif (model_to_train == 1):
        print("You selected to train the pump activating model.")
        train_pump_activating_model(training_data, validation_data, testing_data)
    elif (model_to_train == 2):
        print("You selected to train the light model.")
        train_light_model(training_data, validation_data, testing_data)
    else:
        print("Command not recognized.")
elif (running_mode == 1):
    print("You chose to run the Machine Learning models")
    
    # Loads pump model from .keras file
    pump_waiting_model      = tf.keras.models.load_model(pump_waiting_keras_model_file_location)
    pump_activating_model   = tf.keras.models.load_model(pump_activating_keras_model_file_location)
    print("Pump model summary:\n", pump_waiting_model.summary(), pump_activating_model.summary())

    # Loads light model from .keras file
    light_model = tf.keras.Sequential([
        tf.keras.models.load_model(light_keras_model_file_location),
        tf.keras.layers.Softmax()
    ])
    print("Light model summary:\n", light_model.summary())

    print("Saved")
    last_data_counter = 0
    while True:
        # If the mode is set to automatic machine learning, controls the system based on models
        automatic_mode = read_line_from_commands_file(5)
        if (automatic_mode == 1):
            prediction_queue, data_counter  = read_application_data_from_file(prediction_queue_file_location, is_prediction_queue=True)
            # If a new data has arrived, compute its prediction
            if (data_counter > last_data_counter and data_counter != -1):
                prediction_input        = generate_abstraction_data(prediction_queue, is_prediction_queue=True)
                pump_prediction_input   = prediction_input[:13]
                light_prediction_input  = prediction_input[[0, 6]]

                (pump_waiting_time, pump_activating_time, light_signal, light_confidence_level) = predict_system_output(
                    pump_waiting_model,
                    pump_prediction_input,
                    pump_activating_model,
                    pump_prediction_input,
                    light_model,
                    light_prediction_input
                )
                send_predicted_signals(pump_waiting_time, pump_activating_time, light_signal, light_confidence_level)
                print(f"Pump waiting time: {pump_waiting_time}s | Pump activating time: {pump_activating_time}s \nLight: {light_signal} | Confidence Level: {light_confidence_level}%")
                last_data_counter = data_counter

else:
    print("Command not recognized.")