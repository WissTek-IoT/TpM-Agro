import serial
from enum import Enum

PACKET_BYTES     = 52
MY_ID            = 1
SERIAL_BAUD_RATE = 9600

class packet_indexes(Enum):
    # Physical Layer
    UPLINK_RSSI     = 0,
    UPLINK_QI       = 1,
    DOWNLINK_RSSI   = 2,
    DOWNLINK_QI     = 3,

    # MAC Layer
    MAC_COUNTER_MSB = 4, 
    MAC_COUNTER_LSB = 5,
    MAC3            = 6,
    MAC4            = 7,

    # Network Layer
    RECEIVER_ID     = 8,
    NET2            = 9,
    TRANSMITTER_ID  = 10,
    NET4            = 11,

    # Transport Layer
    DL_COUNTER_MSB = 12,
    DL_COUNTER_LSB = 13,
    UL_COUNTER_MSB = 14,
    UL_COUNTER_LSB = 15,

    # Application Layer
    TEMPERATURE_BYTE_0      = 16,
    TEMPERATURE_BYTE_1      = 17,
    HUMIDITY_BYTE_0         = 18,
    HUMIDITY_BYTE_1         = 19,
    VISIBLE_LIGHT_BYTE_0    = 20,
    VISIBLE_LIGHT_BYTE_1    = 21,
    IR_LIGHT_BYTE_0         = 22,
    IR_LIGHT_BYTE_1         = 23,
    UV_INDEX_BYTE_0         = 24,
    UV_INDEX_BYTE_1         = 25,
    CONTROL_TYPE_INDEX      = 26,
    APP12 = 27,
    APP13 = 28, 
    APP14 = 29,
    APP15 = 30,
    APP16 = 31,
    APP17 = 32,
    APP18 = 33,
    APP19 = 34,
    APP20 = 35,
    APP21 = 36,
    APP22 = 37,
    APP23 = 38,
    APP24 = 39,
    APP25 = 40,
    APP26 = 41,
    APP27 = 42,
    APP28 = 43,
    APP29 = 44,
    APP30 = 45,
    APP31 = 46,
    APP32 = 47,
    APP33 = 48,
    APP34 = 49,
    APP35 = 50,
    APP36 = 51,

serial_port = "COM6"

dl_packet = [0]*PACKET_BYTES
ul_packet = [0]*PACKET_BYTES

# Init Serial communication
while True:
    try:
        ser = serial.Serial(serial_port, SERIAL_BAUD_RATE, timeout = 0.5)
        break
    except:
        pass
print("Conexão Serial estabelecida na porta " + serial_port + ".")
ser.reset_input_buffer()
ser.reset_output_buffer()

while True:
# Receive packet
    ul_packet = ser.read(PACKET_BYTES)

    if len(ul_packet) == PACKET_BYTES:
        for i in range(PACKET_BYTES):
            print("[{}]: {}".format(i, ul_packet[i]))
        ser.reset_input_buffer()
        print()
