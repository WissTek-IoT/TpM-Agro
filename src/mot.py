import serial
import time
from enum import Enum

PACKET_BYTES     = 52
MY_ID            = 1
RECEIVER_ID      = 11
SERIAL_BAUD_RATE = 9600

temperature             = 0
humidity                = 0
visible_light_intensity = 0
ir_light_intensity      = 0
uv_index                = 0
control_mode            = 0

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
    APP12 = 27
    APP13 = 28 
    APP14 = 29
    APP15 = 30
    APP16 = 31
    APP17 = 32
    APP18 = 33
    APP19 = 34
    APP20 = 35
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

serial_port = "COM6"

dl_packet = [0]*PACKET_BYTES
ul_packet = [0]*PACKET_BYTES

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

# Sends the dl packet to the device
def send_packet():
    ser.reset_input_buffer()

    dl_packet[packet_indexes.TRANSMITTER_ID.value]  = MY_ID
    dl_packet[packet_indexes.RECEIVER_ID.value]     = RECEIVER_ID

    ser.write(dl_packet)
    ser.flush()

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

    if debug:     
        print(("Temperature: {} °C\n" +
                "Humidity: {}%\n" +
                "Visible: {} lm | IR: {} lm | UV Index: {}\n" +
                "Control Mode: {}\n").format(
                    temperature,
                    humidity,
                    visible_light_intensity,
                    ir_light_intensity,
                    uv_index,
                    control_mode
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

    # Extract application data from packet
    temperature             = read_value_from_packet(packet_indexes.TEMPERATURE_BYTE_0.value,   True)
    humidity                = read_value_from_packet(packet_indexes.HUMIDITY_BYTE_0.value,      True)

    visible_light_intensity = read_value_from_packet(packet_indexes.VISIBLE_LIGHT_BYTE_0.value, False)
    ir_light_intensity      = read_value_from_packet(packet_indexes.IR_LIGHT_BYTE_0.value,      False)
    uv_index                = read_value_from_packet(packet_indexes.UV_INDEX_BYTE_0.value,      True)

    control_mode            = read_value_from_packet(packet_indexes.CONTROL_TYPE_INDEX.value,   False, True)

    debug_application_data(True)

# Main code (loop)
while True:
    send_packet()
    time.sleep(0.1)     # Waits for the ul packet
    extract_packet()    # Removes the ini and the end\n portions of the packet
    time.sleep(0.5)     # interval between packets

    if len(ul_packet) == PACKET_BYTES:
        debug_received_packet(False)
        read_application_packet()
    else:
        print("Perdeu pacote")
        ser.reset_input_buffer()