from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from enum import Enum
import numpy as np
import tensorflow as tf

class data_indexes(Enum):
    DATE_INDEX          = 0
    HOUR_INDEX          = 1

    TEMPERATURE_INDEX   = 2
    HUMIDITY_INDEX      = 3

    VISIBLE_LIGHT_INDEX = 4
    IR_LIGHT_INDEX      = 5
    UV_INDEX            = 6

    CONTROL_MODE_INDEX  = 7

    PUMP_ENABLED_INDEX  = 8
    LIGHT_ENABLED_INDEX = 9

# Files
application_data_file_location  = os.path.join(os.path.dirname(__file__), '../L4_Storage/application_data.txt')
abstraction_data_file_location  = os.path.join(os.path.dirname(__file__), '../L4_Storage/abstraction_data.txt')
commands_file_location          = os.path.join(os.path.dirname(__file__), '../L4_Storage/commands.txt')
training_data_file_location     = os.path.join(os.path.dirname(__file__), '../L4_Storage/training_data.txt')
validation_data_file_location   = os.path.join(os.path.dirname(__file__), '../L4_Storage/validation_data.txt')
prediction_queue_file_location  = os.path.join(os.path.dirname(__file__), '../L4_Storage/prediction_queue.txt')
model_file_location             = 'L4_Storage/model.keras'

# Application variables
date                = []
hour                = []
temperature         = []
humidity            = []
visible_light       = []
ir_light            = []
uv_index            = []
control_mode        = []
output_label        = []
pump_enabled        = []
light_enabled       = []
automatic_mode_type = 0

# Abstraction variables
hour_in_seconds   = []
time_interval     = []
elapsed_time      = []
d_temperature     = []
d_humidity        = []
d_visible_light   = []
d_ir_light        = []
d_uv_index        = []
last_data_counter = 0

# Datetime variables
time_format = "%d-%m-%Y;%H:%M:%S"

# Threshold for Outliers
temperature_outlier     = 40.0
humidity_outlier        = 100.1
visible_light_outlier   = 20000
ir_light_outlier        = 20000
uv_index_outlier        = 15.0
control_mode_outlier    = 2

# Functions

    # Data Processing Functions
def get_seconds(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def read_automatic_mode_type():
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
    if (len(commands) >= 5):
        return int(commands[5])
    
    return 0

def store_application_data(data_line):
    global date            
    global hour           
    global temperature     
    global humidity        
    global visible_light   
    global ir_light        
    global uv_index        
    global control_mode  
    global output_label
    global pump_enabled
    global light_enabled

    # Stores each data value into a temporary variable
    current_date = data_line[data_indexes.DATE_INDEX.value]
    current_hour = data_line[data_indexes.HOUR_INDEX.value]
    current_temperature     = float   (data_line[data_indexes.TEMPERATURE_INDEX   .value])
    current_humidity        = float   (data_line[data_indexes.HUMIDITY_INDEX      .value])
    current_visible_light   = int     (data_line[data_indexes.VISIBLE_LIGHT_INDEX .value])
    current_ir_light        = int     (data_line[data_indexes.IR_LIGHT_INDEX      .value])
    current_uv_index        = float   (data_line[data_indexes.UV_INDEX            .value])
    current_control_mode    = int     (data_line[data_indexes.CONTROL_MODE_INDEX  .value])
    current_pump_enabled    = int     (data_line[data_indexes.PUMP_ENABLED_INDEX  .value])
    current_light_enabled   = int     (data_line[data_indexes.LIGHT_ENABLED_INDEX .value])
    current_output_label    =         (current_pump_enabled + (current_light_enabled << 1)) # Combine output variables to form a single label, useful for the ML algorithm

    # If none of the values is an outlier, store the data line
    if (
        current_temperature     < temperature_outlier   and
        current_humidity        < humidity_outlier      and
        current_visible_light   < visible_light_outlier and
        current_ir_light        < ir_light_outlier      and
        current_uv_index        < uv_index_outlier      and
        current_control_mode    < control_mode_outlier
    ):
        # Stores data into respective arrays
        date            .append(current_date)
        hour            .append(current_hour)
        temperature     .append(current_temperature)
        humidity        .append(current_humidity)
        visible_light   .append(current_visible_light)
        ir_light        .append(current_ir_light)
        uv_index        .append(current_uv_index)
        control_mode    .append(current_control_mode)
        pump_enabled    .append(current_pump_enabled)
        light_enabled   .append(current_light_enabled)
        output_label    .append(current_output_label)

def read_application_data():
    global date            
    global hour           
    global temperature     
    global humidity        
    global visible_light   
    global ir_light        
    global uv_index        
    global control_mode    
    global output_label
    global pump_enabled    
    global light_enabled    

    application_data_file = open(application_data_file_location, 'r', encoding='utf-8-sig')

    for line in application_data_file:
        # Separates data from semicolon
        line = line.strip()
        data_line = line.split(';')

        store_application_data(data_line)

def compute_abstraction_data():
    global date            
    global hour           
    global temperature     
    global humidity        
    global visible_light   
    global ir_light        
    global uv_index        
    global control_mode  
    global output_label
    global pump_enabled 
    global light_enabled

    global hour_in_seconds
    global time_interval        
    global elapsed_time           
    global d_temperature     
    global d_humidity        
    global d_visible_light   
    global d_ir_light        
    global d_uv_index         

    for i in range(len(date)):
        hour_in_seconds.append(get_seconds(hour[i]))

        if (i != 0):
            # Calculate elapsed time
            initial_time    = datetime.strptime(str(date[0] + ';' + hour[0])    ,time_format)   # Merges separated date and hour presented in time_format format into a datetime instance
            current_time    = datetime.strptime(str(date[i] + ';' + hour[i])    ,time_format)   # Same as above
            elapsed_time.append((current_time-initial_time).total_seconds())

            # Calculate time interval
            previous_time   = datetime.strptime(str(date[i-1] + ';' + hour[i-1]), time_format)  # Same as above
            delta_seconds = (current_time - previous_time).total_seconds()
            time_interval.append(int(delta_seconds))

            # Calculate derivatives
            d_temperature       .append((temperature[i]     - temperature[i-1])     /time_interval[i])
            d_humidity          .append((humidity[i]        - humidity[i-1])        /time_interval[i])
            d_visible_light     .append((visible_light[i]   - visible_light[i-1])   /time_interval[i])
            d_ir_light          .append((ir_light[i]        - ir_light[i-1])        /time_interval[i])
            d_uv_index          .append((uv_index[i]        - uv_index[i-1])        /time_interval[i])
        else :
            # We cannot calculate derivatives for the 1st data line
            elapsed_time        .append(0)
            time_interval       .append(9999999)
            d_temperature       .append(0)
            d_humidity          .append(0)
            d_visible_light     .append(0)
            d_ir_light          .append(0)
            d_uv_index          .append(0)

def store_abstraction_data():
    abstraction_data_file = open(abstraction_data_file_location, 'w+')
    if (abstraction_data_file.writable()):
        for i in range(len(date)):
            data_to_write = [
                # date[i],
                # hour[i],
                hour_in_seconds[i],
                temperature[i],
                humidity[i],
                visible_light[i],
                ir_light[i],
                uv_index[i],
                # control_mode[i],
                elapsed_time[i],
                time_interval[i],
                d_temperature[i],
                d_humidity[i],
                d_visible_light[i],
                d_ir_light[i],
                d_uv_index[i],
                output_label[i],
                # pump_enabled[i],
                # light_enabled[i],
                "\n"
            ]
            
            # Converts all data to string and then separate it by semicolons
            temp = list(map(str, data_to_write))
            res = ";".join(temp)

            # Stores data into text file
            abstraction_data_file.write(res)
    abstraction_data_file.close()

def store_training_data(final_index):
    training_data_file = open(training_data_file_location, 'w+')
    if (training_data_file.writable()):
        for i in range(final_index):
            data_to_write = [
                # date[i],
                # hour[i],
                hour_in_seconds[i],
                temperature[i],
                humidity[i],
                visible_light[i],
                ir_light[i],
                uv_index[i],
                # control_mode[i],
                elapsed_time[i],
                time_interval[i],
                d_temperature[i],
                d_humidity[i],
                d_visible_light[i],
                d_ir_light[i],
                d_uv_index[i],
                output_label[i],
                # pump_enabled[i],
                # light_enabled[i],
                "\n"
            ]
            
            # Converts all data to string and then separate it by semicolons
            temp = list(map(str, data_to_write))
            res = ";".join(temp)

            # Stores data into text file
            training_data_file.write(res)
    training_data_file.close()

def store_validation_data(starting_index):
    validation_data_file = open(validation_data_file_location, 'w+')
    if (validation_data_file.writable()):
        for i in range(starting_index, len(date)):
            data_to_write = [
                # date[i],
                # hour[i],
                hour_in_seconds[i],
                temperature[i],
                humidity[i],
                visible_light[i],
                ir_light[i],
                uv_index[i],
                # control_mode[i],
                elapsed_time[i],
                time_interval[i],
                d_temperature[i],
                d_humidity[i],
                d_visible_light[i],
                d_ir_light[i],
                d_uv_index[i],
                output_label[i],
                # pump_enabled[i],
                # light_enabled[i],
                "\n"
            ]
            
            # Converts all data to string and then separate it by semicolons
            temp = list(map(str, data_to_write))
            res = ";".join(temp)

            # Stores data into text file
            validation_data_file.write(res)
    validation_data_file.close()

    # Machine Learning Functions
def read_training_data():
    X = []
    Y = []

    # Open training data file and stores its inputs and respective outputs
    training_data_file = open(training_data_file_location, 'r', encoding='utf-8-sig')
    for line in training_data_file:
        # Separates data from semicolon
        line = line.strip()
        data_line = line.split(';')

        # Separates input (X) and output (Y) data
        X.append(data_line[:13])
        Y.append(data_line[-2:-1])

    # Converts data into a numpy array
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(int).flatten()

    # print(( "Input Shape: {} \n"+
    #         "Output Shape: {}").format(X.shape, Y.shape))
    
    return X, Y

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
        X.append(data_line[:13])
        Y.append(data_line[-2:-1])

    # Converts data into a numpy array
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(int).flatten()
    
    return X, Y

def read_prediction_queue():
    data_in_file    = []

    prediction_queue_file = open(prediction_queue_file_location, 'r', encoding='utf-8-sig')
    for line in prediction_queue_file:
        # Separates data from semicolon
        line = line.strip()
        data_line = line.split(';')
        data_in_file.append(data_line)

    for i in range(len(data_in_file)):
        if i > 0:
            data_in_file[i][data_indexes.TEMPERATURE_INDEX.value]   = float (data_in_file[i][data_indexes.TEMPERATURE_INDEX.value])
            data_in_file[i][data_indexes.HUMIDITY_INDEX.value]      = float (data_in_file[i][data_indexes.HUMIDITY_INDEX.value])
            data_in_file[i][data_indexes.VISIBLE_LIGHT_INDEX.value] = int   (data_in_file[i][data_indexes.VISIBLE_LIGHT_INDEX.value])
            data_in_file[i][data_indexes.IR_LIGHT_INDEX.value]      = int   (data_in_file[i][data_indexes.IR_LIGHT_INDEX.value])
            data_in_file[i][data_indexes.UV_INDEX.value]            = float (data_in_file[i][data_indexes.UV_INDEX.value])
            data_in_file[i][data_indexes.PUMP_ENABLED_INDEX.value]  = int   (data_in_file[i][data_indexes.PUMP_ENABLED_INDEX.value])
            data_in_file[i][data_indexes.LIGHT_ENABLED_INDEX.value] = int   (data_in_file[i][data_indexes.LIGHT_ENABLED_INDEX.value])
            data_in_file[i][data_indexes.VISIBLE_LIGHT_INDEX.value] = int   (data_in_file[i][data_indexes.VISIBLE_LIGHT_INDEX.value])
        else:
            data_in_file[i][i] = int(data_in_file[i][i])

    return data_in_file

def compute_abstraction_for_prediction_queue(data_in_file):
    global last_data_counter
    input_for_prediction = []

    # If list is not empty, compute data
    if data_in_file:
        data_counter            = data_in_file[0][0]

        initial_date            = data_in_file[1][0]
        initial_hour            = data_in_file[1][1]

        previous_date           = data_in_file[2][0]
        previous_hour           = data_in_file[2][1]
        previous_temperature    = data_in_file[2][2]
        previous_humidity       = data_in_file[2][3]
        previous_visible_light  = data_in_file[2][4]
        previous_ir_light       = data_in_file[2][5]
        previous_uv_index       = data_in_file[2][6]

        current_date            = data_in_file[3][0]
        current_hour            = data_in_file[3][1]
        current_temperature     = data_in_file[3][2]
        current_humidity        = data_in_file[3][3]
        current_visible_light   = data_in_file[3][4]
        current_ir_light        = data_in_file[3][5]
        current_uv_index        = data_in_file[3][6]

        # Only go forward if new data is received
        if (data_counter > last_data_counter):
            # Calculate elapsed time
            initial_time = datetime.strptime(str(initial_date + ";" + initial_hour), time_format)
            current_time = datetime.strptime(str(current_date + ";" + current_hour), time_format)
            elapsed_time = (current_time - initial_time).total_seconds()
            current_hour_in_seconds = get_seconds(current_hour)

            # Calculate time interval
            previous_time = datetime.strptime(str(previous_date + ";" + previous_hour), time_format)
            delta_time    = (current_time - previous_time).total_seconds()

            # Calculate derivatives
            d_temperature   =   (current_temperature    - previous_temperature)     /delta_time
            d_humidity      =   (current_humidity       - previous_humidity)        /delta_time
            d_visible_light =   (current_visible_light  - previous_visible_light)   /delta_time
            d_ir_light      =   (current_ir_light       - previous_ir_light)        /delta_time
            d_uv_index      =   (current_uv_index       - previous_uv_index)        /delta_time

            # After processing data, assemble input packet
            input_for_prediction = [
                current_hour_in_seconds,
                current_temperature,
                current_humidity,
                current_visible_light,
                current_ir_light,
                current_uv_index,
                elapsed_time,
                delta_time,
                d_temperature,
                d_humidity,
                d_visible_light,
                d_ir_light,
                d_uv_index,
            ]
            last_data_counter = data_counter
            input_for_prediction = np.array(input_for_prediction).astype(float)

            return True, input_for_prediction
    
    return False, -1

def predict_system_output(input_for_prediction, model):
    input = np.expand_dims(input_for_prediction, 0)

    prediction          = model.predict(input)
    prediction_label    = np.argmax(prediction)

    # Extracts the predicted value from the predicted label
    pump_signal         = prediction_label & 0x01
    light_signal        = prediction_label >> 1
    confidence_level    = round(prediction[0][prediction_label]*100, 2)

    return pump_signal, light_signal, confidence_level

# Main Code
user_input = int(input("Select between: Train ML Model (0) | Run ML Model (1)\n"))

if (user_input == 0):
    print("\nYou chose to train the Machine Learning Model.")

    print("Computing abstraction data...")
    read_application_data()
    compute_abstraction_data()
    store_abstraction_data()
    print("Abstraction data processed.\n")

    total_data_length       = len(date)
    training_data_length    = round(0.7*total_data_length)
    validation_data_length  = total_data_length - training_data_length
    print(f"Total data: {total_data_length} lines")
    print(f"70% of total data ({training_data_length} lines) will be used to train the model")
    print(f"30% of total data ({validation_data_length} lines) will be used to validate the model")
    store_training_data(training_data_length)
    store_validation_data(training_data_length)

    # Reads data
    X_train, Y_train = read_training_data()
    X_valid, Y_valid = read_validation_data()
    # Normalize input values X
    normalization = tf.keras.layers.Normalization(axis=-1)  # Creates a normalization instance
    normalization.adapt(X_train)                            # Learns mean and variance from input dataset X
    # Creates the Machine Learning Model
    model = tf.keras.Sequential([
        normalization,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
    ])
    print("\nPerforming trainning...")
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=10)
    print("Finished training.")
    print(model.summary())

    print("\nValidating model...")
    valid_loss, valid_accuracy = model.evaluate(X_valid,  Y_valid, verbose=0)

    # Computes predicted output for each validation set's input
    probability_model               = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions_for_validation_set  = probability_model.predict(X_valid)
    print(f"Accuracy: {round(100*valid_accuracy,3)}%")
    print(f"Loss: {round(100*valid_loss,3)}%")

    model.save(model_file_location)
    print("\nModel was saved on L4_Storage")

elif (user_input == 1):
    model = tf.keras.models.load_model(model_file_location)
    print("You chose to run the Machine Learning model below:")
    print(model.summary())
    prediction_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    while True:
        automatic_mode_type = read_automatic_mode_type()
        if (automatic_mode_type == 1):
            prediction_queue                    = read_prediction_queue()
            valid_input, input_for_predicton    = compute_abstraction_for_prediction_queue(prediction_queue)
            if (valid_input):
                pump_signal, light_signal, confidence_level = predict_system_output(input_for_predicton, prediction_model)
                print(f"Pump: {pump_signal} | Light: {light_signal}\nConfidence Level: {confidence_level}%")

else:
    print("Command not recognized.")

