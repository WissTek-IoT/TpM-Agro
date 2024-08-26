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

    PUMP_ENABLED_INDEX  = 8
    LIGHT_ENABLED_INDEX = 9

# Files
application_data_file_location  = os.path.join(os.path.dirname(__file__), '../NIVEL_4/test_data.txt')

abstraction_data_file_location  = os.path.join(os.path.dirname(__file__), '../NIVEL_4/abstraction_data.txt')
abstraction_data_file           = open(abstraction_data_file_location, 'a+')
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
output_label    = []
pump_enabled    = []
light_enabled   = []

# Abstraction arrays
hour_in_seconds   = []
time_interval     = []
elapsed_time      = []
d_temperature     = []
d_humidity        = []
d_visible_light   = []
d_ir_light        = []
d_uv_index        = []

# Datetime variables
time_format = "%d-%m-%Y;%H:%M:%S"

# Threshold for Outliers
temperature_outlier     = 40.0
humidity_outlier        = 100.1
visible_light_outlier   = 20000
ir_light_outlier        = 20000
uv_index_outlier        = 15.0
control_mode_outlier    = 2

def get_seconds(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

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
    abstraction_data_file = open(abstraction_data_file_location, 'a')
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
                control_mode[i],
                elapsed_time[i],
                time_interval[i],
                d_temperature[i],
                d_humidity[i],
                d_visible_light[i],
                d_ir_light[i],
                d_uv_index[i],
                output_label[i],
                pump_enabled[i],
                light_enabled[i],
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