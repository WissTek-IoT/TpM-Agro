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

def debug_received_packet(debug):
    if (debug):
        for i in range(PACKET_BYTES):
            print("[{}]: {}".format(i, ul_packet[i]))
        ser.reset_input_buffer()
        print()

def read_value_from_packet(initial_index, is_float, is_single_byte = False):
    data = 0
    multiplier = 256

    if (is_single_byte): multiplier = 0
    
    data = (ul_packet[initial_index] + 
            ul_packet[initial_index + 1] * multiplier)
    
    if (is_float): data /= 10

    return data

def send_packet():
    dl_packet[packet_indexes.TRANSMITTER_ID.value]  = MY_ID
    dl_packet[packet_indexes.RECEIVER_ID.value]     = RECEIVER_ID

    ser.write(dl_packet)
    ser.flush()

def find_ini_sequence():
    ini_first_i_index   = ul_packet.find(ord('i'))
    ini_n_index         = ul_packet.find(ord('n'))
    ini_last_i_index    = ul_packet.find(ord('i'), ini_first_i_index + 1)

    ini_index           = [ini_first_i_index, ini_n_index, ini_last_i_index]
    return ini_index

def find_end_sequence():
    end_e_index     = ul_packet.rfind(ord('e'))
    end_n_index     = ul_packet.rfind(ord('n'))
    end_d_index     = ul_packet.rfind(ord('d'))
    end_LF_index    = ul_packet.rfind(ord('\n'))

    end_index           = [end_e_index, end_n_index, end_d_index, end_LF_index]
    return end_index

def is_sequential(list):
    return all(list[i] == list[i-1] + 1 for i in range(1, len(list)))

while True:
    ser.reset_input_buffer()
    send_packet()
    time.sleep(1.0)

    ul_packet = ser.readline()
    # print(ul_packet.rfind(ord('\n')))
    ini_first_i_index   = ul_packet.find(ord('i'))
    ini_n_index         = ul_packet.find(ord('n'))
    ini_last_i_index    = ul_packet.find(ord('i'), ini_first_i_index + 1)

    ini_index           = [ini_first_i_index, ini_n_index, ini_last_i_index]
    is_ini_sequential   = all(ini_index[i] == ini_index[i-1] + 1 for i in range(1, len(ini_index)))

    if is_ini_sequential:
        end_e_index     = ul_packet.rfind(ord('e'))
        end_n_index     = ul_packet.rfind(ord('n'))
        end_d_index     = ul_packet.rfind(ord('d'))
        end_LF_index    = ul_packet.rfind(ord('\n'))

        end_index           = [end_e_index, end_n_index, end_d_index, end_LF_index]
        is_end_sequential   = all(end_index[i] == end_index[i-1] + 1 for i in range(1, len(end_index)))
        # print(end_index)

        if is_end_sequential:
            ul_packet = ul_packet[(ini_index[2]+1):end_index[0]]
    # ul_packet = ser.readline()
    # end_of_packet = ul_packet.find(ord('\n'))
    # print(end_of_packet)
    # ul_packet = ul_packet[:end_of_packet]
    # ul_packet = ul_packet[-52:]
    # print(ul_packet)
    # print("L: ", len(ul_packet))

    # teste = ser.readline()
    # index_teste = teste.find(ord('\n'))
    # new_teste = teste[:index_teste]
    # new_teste = new_teste[-2:]
    # print(index_teste)
    # print(new_teste)
    # print("L: ", len(new_teste))
    # ini_character_pos = [teste.find(ord('i')), teste.find(ord('n')), teste.find(ord('i'), 1)]
    # is_consecutive = all(ini_character_pos[i] == ini_character_pos[i-1] + 1 for i in range(1, len(ini_character_pos)))
    # print("L:", len(teste))
    # print(teste.find(105))
    # print(teste.find(110))
    # print(teste.find(105,1))
    # print(ini_character_pos)
    # print(is_consecutive)
    # print(teste)
    # pos_ini = teste.find("ini")
    # pos_end = teste.find("end")
    # sub_teste = teste[pos_ini+3:pos_end]
    # print("L: ", len(sub_teste))
    # print(sub_teste)
    # ul_packet = sub_teste



    # print(ser.readline())
# Receive packet
    # send_packet()
    # time.sleep(1.0) # waits for the dl packet (time taken by arduino to finish writing the dl packet)
    # # ul_packet = ser.read(PACKET_BYTES)
    # ul_packet = ser.readline(PACKET_BYTES)
    # # ser.reset_input_buffer()
    # print(len(ul_packet))
    if len(ul_packet) == PACKET_BYTES:
        debug_received_packet(False)

        # Extract data from packet
        temperature             = read_value_from_packet(packet_indexes.TEMPERATURE_BYTE_0.value,   True)
        humidity                = read_value_from_packet(packet_indexes.HUMIDITY_BYTE_0.value,      True)

        visible_light_intensity = read_value_from_packet(packet_indexes.VISIBLE_LIGHT_BYTE_0.value, False)
        ir_light_intensity      = read_value_from_packet(packet_indexes.IR_LIGHT_BYTE_0.value,      False)
        uv_index                = read_value_from_packet(packet_indexes.UV_INDEX_BYTE_0.value,      True)

        control_mode            = read_value_from_packet(packet_indexes.CONTROL_TYPE_INDEX.value,   False, True)

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
    else:
        print("Perdeu pacote")
        ser.reset_input_buffer()

    # time.sleep(1.0)
