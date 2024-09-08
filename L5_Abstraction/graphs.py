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
import matplotlib.pyplot as plt

# FILES
application_data_file_location      = os.path.join(os.path.dirname(__file__), '../L4_Storage/application_data.txt')
abstraction_data_file_location      = os.path.join(os.path.dirname(__file__), '../L4_Storage/abstraction_data.txt')
commands_file_location              = os.path.join(os.path.dirname(__file__), '../L4_Storage/commands.txt')
prediction_queue_file_location      = os.path.join(os.path.dirname(__file__), '../L4_Storage/prediction_queue.txt')
pump_waiting_model_file_location    = "L4_Storage/pump_waiting_model.keras"
pump_activating_model_file_location = "L4_Storage/pump_ativating_model.keras"
light_model_file_location           = "L4_Storage/light_model.keras"

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

# "NORMALIZE" OUTPUTS FOR REGRESSION MODEL
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

def read_data_file(file_location, is_prediction_queue=False):
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

application_data = read_data_file(application_data_file_location)
abstraction_data = generate_abstraction_data(application_data)

pump_waiting_model      = tf.keras.models.load_model(pump_waiting_model_file_location)
pump_activating_model   = tf.keras.models.load_model(pump_activating_model_file_location)
light_model = tf.keras.Sequential([
        tf.keras.models.load_model(light_model_file_location),
        tf.keras.layers.Softmax()
])

# pump_waiting_model.evaluate(abstraction_data[192257:, 0:13], abstraction_data[192257:, 14])
pump_w_pred = pump_waiting_model.predict(abstraction_data[:, 0:13])
pump_a_pred = pump_activating_model.predict(abstraction_data[:, 0:13])
light_pred = light_model.predict(abstraction_data[:, [0, 6]])
light_o = []
for i in range(len(light_pred)):
    output = np.argmax(light_pred[i])
    light_o.append(output)

print(pump_w_pred[0])
print(pump_a_pred[0])
print(light_o[0])

# plot
fig, ax = plt.subplots()
# size and color:
ax.scatter(abstraction_data[:, 6]/3600, (np.round(100*pump_w_pred,0)/60,),alpha=0.2, c="#2adb6b", s=5)
ax.scatter(abstraction_data[:, 6]/3600, (100*abstraction_data[:, 14])/60, c="black", s=5)
ax.scatter(abstraction_data[:, 6]/3600, (27*abstraction_data[:, 16]), c="orange", s=5)
ax.set(ylim=(-0.1, max((100*pump_w_pred)/60)+5))
ax.set_xlabel("Tempo decorrido desde o início do cultivo (horas)", fontsize=14)
ax.set_ylabel("Intervalo entre ciclos de acionamento (minutos)", fontsize=14)

leg = plt.legend(["Modelo de Regressão", "Referência", "Luz"], fontsize="10", loc="upper right", markerscale=4)
for lh in leg.legend_handles:
    lh.set_alpha(1)
# plt.legend(["Modelo de Regressão", "Referência", "Luz"], fontsize="10", loc="upper right", markerscale=4)
# ax2 = ax.twinx()
# t3=ax2.scatter(abstraction_data[:, 6]/86400, (abstraction_data[:, 16]), c="orange", s=5)
# ax2.set_ylabel("Estado da luz de crescimento", fontsize=14)
# ax2.set(ylim=(-0.01, 2))
# ax2.legend(["Estado da luz de crescimento"], fontsize="10", loc="upper left", markerscale=4)


plt.show()

fig, ax = plt.subplots()
ax.scatter(abstraction_data[:, 6]/3600, (np.round(10*pump_a_pred)), alpha=0.2, c="#2adb6b", s=12, label="Modelo")
ax.scatter(abstraction_data[:, 6]/3600, (10*abstraction_data[:, 15]), c="black", s=12, label="Real")
ax.set(ylim=(-0.01, max((10*pump_a_pred))+5))
leg = plt.legend(["Modelo de Regressão", "Referência"], fontsize="10", loc="upper right", markerscale=3)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel("Tempo decorrido desde o início do cultivo (horas)", fontsize=14)
plt.ylabel("Duração de um acionamento (segundos)", fontsize=14)
plt.show()

fig, ax = plt.subplots()
ax.scatter(abstraction_data[:, 6]/3600, light_o, c="#2adb6b", s=12, label="Modelo")
ax.scatter(abstraction_data[:, 6]/3600, (abstraction_data[:, 16]), c="black", s=12, label="Real")
ax.set(ylim=(-0.3, 1.5))
plt.legend(["Modelo de Classificação", "Referência"], fontsize="10", loc="upper right", markerscale=3)
plt.xlabel("Tempo decorrido desde o início do cultivo (horas)", fontsize=14)
plt.ylabel("Duração de um acionamento (segundos)", fontsize=14)
plt.show()