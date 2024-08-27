import serial
from enum import Enum
import time
import os

class packet_indexes(Enum):
    # Physical Layer
    UPLINK_RSSI     = 0
    UPLINK_QI       = 1
    DOWNLINK_RSSI   = 2
    DOWNLINK_QI     = 3

    # MAC Layer
    MAC_COUNTER_MSB = 4 
    MAC_COUNTER_LSB = 5
    MAC3            = 6
    MAC4            = 7

    # Network Layer
    RECEIVER_ID     = 8
    NET2            = 9
    TRANSMITTER_ID  = 10
    NET4            = 11

    # Transport Layer
    DL_COUNTER_MSB = 12
    DL_COUNTER_LSB = 13
    UL_COUNTER_MSB = 14
    UL_COUNTER_LSB = 15

    # Application Layer
    TEMPERATURE_BYTE_0      = 16
    TEMPERATURE_BYTE_1      = 17
    HUMIDITY_BYTE_0         = 18
    HUMIDITY_BYTE_1         = 19
    VISIBLE_LIGHT_BYTE_0    = 20
    VISIBLE_LIGHT_BYTE_1    = 21
    IR_LIGHT_BYTE_0         = 22
    IR_LIGHT_BYTE_1         = 23
    UV_INDEX_BYTE_0         = 24
    UV_INDEX_BYTE_1         = 25
    CONTROL_TYPE_INDEX      = 26
    PUMP_SIGNAL             = 27
    LIGHT_SIGNAL            = 28 
    PUMP_INTERVAL_BYTE_0    = 29
    PUMP_INTERVAL_BYTE_1    = 30
    PUMP_DURATION_BYTE_0    = 31
    PUMP_DURATION_BYTE_1    = 32
    AUTOMATIC_MODE_TYPE     = 33
    IS_PUMP_ENABLED         = 34
    IS_LIGHT_ENABLED        = 35
    APP21 = 36
    APP22 = 37
    APP23 = 38
    APP24 = 39
    APP25 = 40
    APP26 = 41
    APP27 = 42
    APP28 = 43
    APP29 = 44
    APP30 = 45
    APP31 = 46
    APP32 = 47
    APP33 = 48
    APP34 = 49
    APP35 = 50
    APP36 = 51

# MoT Parameters and Variables
PACKET_BYTES     = 52
MY_ID            = 1
RECEIVER_ID      = 11
dl_packet = [0]*PACKET_BYTES
ul_packet = [0]*PACKET_BYTES
SERIAL_BAUD_RATE = 9600
serial_port = "COM6"

# Application Data
temperature             = 0
humidity                = 0
visible_light_intensity = 0
ir_light_intensity      = 0
uv_index                = 0
control_mode            = 0
pump_enabled            = 0
light_enabled           = 0

# Border Data
timestamp               = ""
data_counter            = 0
first_dataset           = ""
current_dataset         = ""
previous_dataset        = ""

# Files
    # Creates the application data file if it already not exists
application_data_file_location  = os.path.join(os.path.dirname(__file__), '../L4_Storage/application_data.txt')
data_to_predict_file_location   = os.path.join(os.path.dirname(__file__), '../L4_Storage/data_to_predict.txt')
application_data_file           = open(application_data_file_location, 'a')
application_data_file.close()

    # Commands File
commands_file_location  = os.path.join(os.path.dirname(__file__), '../L4_Storage/commands.txt')

# Commands
communication_interval      = 1.0
pump_signal                 = 0
light_signal                = 0
pump_activation_interval    = 60
pump_activation_duration    = 5
automatic_mode_type         = 0
hour_to_turn_on_light       = "06:00:00"
hour_to_turn_off_light      = "18:00:00"

# Init Serial communication
while True:
    try:
        ser = serial.Serial(serial_port, SERIAL_BAUD_RATE)
        break
    except:
        pass
print("Conexão Serial estabelecida na porta " + serial_port + ".")
ser.reset_input_buffer()
ser.reset_output_buffer()

def get_seconds(time_string):
    h, m, s = time_string.split(':')
    return int(h)*3600 + int(m)*60 + int(s)

def control_light_based_on_time():
    global light_signal

    current_hour    = time.strftime("%H:%M:%S")

    seconds_to_turn_on_light    = get_seconds(hour_to_turn_on_light)
    seconds_to_turn_off_light   = get_seconds(hour_to_turn_off_light)
    current_seconds             = get_seconds(current_hour)
    evaluation = (  current_seconds > seconds_to_turn_on_light  and 
                    current_seconds < seconds_to_turn_off_light)
    
    light_signal = evaluation

def store_command_variables(commands):
    global communication_interval
    global pump_signal
    global light_signal
    global pump_activation_interval
    global pump_activation_duration
    global automatic_mode_type
    global hour_to_turn_on_light
    global hour_to_turn_off_light

    # Stores all commands into its variables
    communication_interval      = int(commands[0])
    pump_signal                 = int(commands[1])
    light_signal                = int(commands[2])
    pump_activation_interval    = int(commands[3])
    pump_activation_duration    = int(commands[4])
    automatic_mode_type         = int(commands[5])
    hour_to_turn_on_light       = str(commands[6])
    hour_to_turn_off_light      = str(commands[7])

    # If the system is in automatic periodic mode, controls light based on current hour
    if (control_mode == 1 and automatic_mode_type == 0):
        control_light_based_on_time()


def read_commands_file():
    commands = []

    commands_file = open(commands_file_location, 'r')
    # Stores all commands into an array
    for line in commands_file:
        # Separates data from colon
        line = line.strip()
        data_line = line.split(';')
        data_line = data_line[1].strip()
        commands.append(data_line)

    store_command_variables(commands)

    commands_file.close()


def assemble_dl_packet():
    global dl_packet
    global automatic_mode_type

    read_commands_file()

    # Network Layer
    dl_packet[packet_indexes.TRANSMITTER_ID.value]          = MY_ID
    dl_packet[packet_indexes.RECEIVER_ID.value]             = RECEIVER_ID

    # Application Layer
    dl_packet[packet_indexes.PUMP_SIGNAL.value]             = pump_signal
    dl_packet[packet_indexes.LIGHT_SIGNAL.value]            = light_signal

    dl_packet[packet_indexes.PUMP_INTERVAL_BYTE_0.value]    = int(pump_activation_interval%256)
    dl_packet[packet_indexes.PUMP_INTERVAL_BYTE_1.value]    = int(pump_activation_interval/256)

    dl_packet[packet_indexes.PUMP_DURATION_BYTE_0.value]    = int(pump_activation_duration%256)
    dl_packet[packet_indexes.PUMP_DURATION_BYTE_1.value]    = int(pump_activation_duration/256)

    dl_packet[packet_indexes.AUTOMATIC_MODE_TYPE.value]     = automatic_mode_type

# Sends the dl packet to the device
def send_packet():
    ser.reset_input_buffer()

    assemble_dl_packet()
    ser.write(dl_packet)
    ser.flush()

# Prints all the content com the received packet
def debug_received_packet(debug):
    if (debug):
        for i in range(PACKET_BYTES):
            print("[{}]: {}".format(i, ul_packet[i]))
        ser.reset_input_buffer()
        print()

# Reads a specific data from the packet
def read_value_from_packet(initial_index, is_float, is_single_byte = False):
    data = 0
    multiplier = 256

    if (is_single_byte): multiplier = 0
    
    data = (ul_packet[initial_index] + 
            ul_packet[initial_index + 1] * multiplier)
    
    if (is_float): data /= 10

    return data

# Find the position of the ini sequence (start of the packet) to avoid discontinuities
def find_ini_sequence():
    ini_first_i_index   = ul_packet.find(ord('i'))
    ini_n_index         = ul_packet.find(ord('n'))
    ini_last_i_index    = ul_packet.find(ord('i'), ini_first_i_index + 1)

    ini_index           = [ini_first_i_index, ini_n_index, ini_last_i_index]
    return ini_index

# Finds the position of the end\n sequence (end of the packet) to avoid discontinuities
def find_end_sequence():
    end_e_index     = ul_packet.rfind(ord('e'))
    end_n_index     = ul_packet.rfind(ord('n'))
    end_d_index     = ul_packet.rfind(ord('d'))
    end_LF_index    = ul_packet.rfind(ord('\n'))

    end_index           = [end_e_index, end_n_index, end_d_index, end_LF_index]
    return end_index

# Evaluate if a given list is composed of sequential numbers
def is_sequential(list):
    return all(list[i] == list[i-1] + 1 for i in range(1, len(list)))

# Extract the data packet (remove the ini and end\n sequences)
def extract_packet():
    global ul_packet

    ul_packet = ser.readline()

    ini_index           = find_ini_sequence()
    ini_sequential      = is_sequential(ini_index)

    end_index           = find_end_sequence()
    end_sequential      = is_sequential(end_index)

    if (ini_sequential and end_sequential):
        ul_packet = ul_packet[(ini_index[2]+1):end_index[0]]

# Prints the information stored into the ul application packet
def debug_application_data(debug):
    global temperature             
    global humidity                
    global visible_light_intensity 
    global ir_light_intensity      
    global uv_index                
    global control_mode
    global pump_enabled
    global light_enabled       

    if debug:     
        print(("Temperature: {} °C\n" +
                "Humidity: {}%\n" +
                "Visible: {} lm | IR: {} lm | UV Index: {}\n" +
                "Control Mode: {}\n" +
                "Pump State: {} | Light State: {}\n").format(
                    temperature,
                    humidity,
                    visible_light_intensity,
                    ir_light_intensity,
                    uv_index,
                    control_mode,
                    pump_enabled,
                    light_enabled
                ))

# Reads the application packet
def read_application_packet():
    global ul_packet
    global temperature             
    global humidity                
    global visible_light_intensity 
    global ir_light_intensity      
    global uv_index                
    global control_mode   
    global pump_enabled
    global light_enabled

    # Extract application data from packet
    temperature             = read_value_from_packet(packet_indexes.TEMPERATURE_BYTE_0.value,   True)
    humidity                = read_value_from_packet(packet_indexes.HUMIDITY_BYTE_0.value,      True)

    visible_light_intensity = read_value_from_packet(packet_indexes.VISIBLE_LIGHT_BYTE_0.value, False)
    ir_light_intensity      = read_value_from_packet(packet_indexes.IR_LIGHT_BYTE_0.value,      False)
    uv_index                = read_value_from_packet(packet_indexes.UV_INDEX_BYTE_0.value,      True)

    control_mode            = read_value_from_packet(packet_indexes.CONTROL_TYPE_INDEX.value,   False, True)

    pump_enabled            = read_value_from_packet(packet_indexes.IS_PUMP_ENABLED.value,     False, True)
    light_enabled           = read_value_from_packet(packet_indexes.IS_LIGHT_ENABLED.value,    False, True)

    debug_application_data(True)

# Stores the application data into the application_data.txt file
def store_application_data():
    global timestamp
    global data_counter
    global first_dataset
    global current_dataset
    global previous_dataset

    application_data_file = open(application_data_file_location, 'a')
    if (application_data_file.writable()):
        timestamp = time.strftime("%d-%m-%Y;%H:%M:%S")

        data_to_write = [
            timestamp,
            temperature,
            humidity,
            visible_light_intensity,
            ir_light_intensity,
            uv_index,
            control_mode,
            pump_enabled,
            light_enabled,
            "\n"
        ]
        
        # Converts all data to string and then separate it by semicolons
        temp = list(map(str, data_to_write))
        res = ";".join(temp)

        # Stores data into text file
        application_data_file.write(res)

        # Saves dataset
        if (data_counter == 0):
            first_dataset = res
        previous_dataset    = current_dataset
        current_dataset     = res
        
        data_counter += 1
    application_data_file.close()

    data_to_predict_file = open(data_to_predict_file_location, 'w+')
    if (data_to_predict_file.writable()):
        data_to_predict_file.write(str(data_counter)+"\n")
        data_to_predict_file.write(first_dataset)
        data_to_predict_file.write(previous_dataset)
        data_to_predict_file.write(current_dataset)
    data_to_predict_file.close()

# Main code (loop)
while True:
    send_packet()
    time.sleep(0.1)     # Waits for the ul packet
    extract_packet()    # Removes the ini and the end\n portions of the packet
    time.sleep(communication_interval)     # interval between packets

    if len(ul_packet) == PACKET_BYTES:
        debug_received_packet(False)
        read_application_packet()
        store_application_data()
    else:
        print("Perdeu pacote")
        ser.reset_input_buffer()