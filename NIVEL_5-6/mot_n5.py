from datetime import datetime
import os
from enum import Enum

class data_indexes(Enum):
    DATE_INDEX          = 0
    HOUR_INDEX          = 1

    TEMPERATURE_INDEX   = 2
    HUMIDITY_INDEX      = 3

    VISIBLE_LIGHT_INDEX = 4
    IR_LIGHT_INDEX      = 5
    UV_INDEX            = 6

    CONTROL_MODE_INDEX  = 7

# Files
application_data_file_location  = os.path.join(os.path.dirname(__file__), '../NIVEL_4/application_data.txt')

abstraction_data_file_location  = os.path.join(os.path.dirname(__file__), '../NIVEL_4/abstraction_data.txt')
abstraction_data_file           = open(abstraction_data_file_location, 'a')
abstraction_data_file.close()

# Data arrays
date            = []
hour            = []
temperature     = []
humidity        = []
visible_light   = []
ir_light        = []
uv_index        = []
control_mode    = []

# Abstraction arrays
time_interval     = []
elapsed_time      = []
d_temperature     = []
d_humidity        = []
d_visible_light   = []
d_ir_light        = []
d_uv_index        = []

# Datetime variables
time_format = "%d-%m-%Y;%H:%M:%S"

def store_application_data(data_line):
    global date            
    global hour           
    global temperature     
    global humidity        
    global visible_light   
    global ir_light        
    global uv_index        
    global control_mode  

    # Stores data into respective arrays
    date            .append(data_line[data_indexes.DATE_INDEX.value])
    hour            .append(data_line[data_indexes.HOUR_INDEX.value])
    temperature     .append(float(data_line[data_indexes.TEMPERATURE_INDEX.value]))
    humidity        .append(float(data_line[data_indexes.HUMIDITY_INDEX.value]))
    visible_light   .append(int(data_line[data_indexes.VISIBLE_LIGHT_INDEX.value]))
    ir_light        .append(int(data_line[data_indexes.IR_LIGHT_INDEX.value]))
    uv_index        .append(float(data_line[data_indexes.UV_INDEX.value]))
    control_mode    .append(int(data_line[data_indexes.CONTROL_MODE_INDEX.value]))

def read_application_data():
    global date            
    global hour           
    global temperature     
    global humidity        
    global visible_light   
    global ir_light        
    global uv_index        
    global control_mode    

    application_data_file = open(application_data_file_location, 'r')

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

    global time_interval        
    global elapsed_time           
    global d_temperature     
    global d_humidity        
    global d_visible_light   
    global d_ir_light        
    global d_uv_index         

    for i in range(len(date)):
        if (i != 0):
            # Calculate elapsed time
            initial_time    = datetime.strptime(str(date[0] + ';' + hour[0]), time_format)
            current_time    = datetime.strptime(str(date[i] + ';' + hour[i])    , time_format)
            elapsed_time.append((current_time-initial_time).total_seconds())

            # Calculate time interval
            previous_time   = datetime.strptime(str(date[i-1] + ';' + hour[i-1]), time_format)
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

    print(d_ir_light)

def store_abstraction_data():
    abstraction_data_file = open(abstraction_data_file_location, 'a')
    if (abstraction_data_file.writable()):
        for i in range(len(date)):
            data_to_write = [
                date[i],
                hour[i],
                temperature[i],
                humidity[i],
                visible_light[i],
                ir_light[i],
                uv_index[i],
                control_mode[i],
                elapsed_time[i],
                time_interval[i],
                d_temperature[i],
                d_humidity[i],
                d_visible_light[i],
                d_ir_light[i],
                d_uv_index[i],
                "\n"
            ]
            
            # Converts all data to string and then separate it by semicolons
            temp = list(map(str, data_to_write))
            res = ";".join(temp)

            # Stores data into text file
            abstraction_data_file.write(res)
    abstraction_data_file.close()

read_application_data()
compute_abstraction_data()
store_abstraction_data()