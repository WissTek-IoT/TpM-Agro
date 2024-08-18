#ifndef __MOT_H__
#define __MOT_H__

#include "utils.h"
#include "pump.h"
#include "light.h"
#include "WTK_TH.h"
#include "mode.h"

// MACROS
    #define PACKET_BYTES        52
    #define MY_ID               11
    #define SERIAL_BAUD_RATE    9600

    #define AUTOMATIC_PERIODIC_MODE 0
    #define AUTOMATIC_ML_MODE       1

// GLOBAL VARIABLES
extern byte dl_packet[PACKET_BYTES];
extern byte ul_packet[PACKET_BYTES];
extern uint32_t dl_packet_counter;
extern uint32_t ul_packet_counter;

// FUNCTIONS
    // MoT
void init_mot_protocol();
void receive_mot_packet();
void send_mot_packet();

    // Physical Layer
void init_physical_layer();
void read_physical_layer_packet();
void assemble_physical_layer_packet();

    // MAC Layer
void init_mac_layer();
void read_mac_layer_packet();
void assemble_mac_layer_packet();

    // Network Layer
void init_network_layer();
void read_network_layer_packet();
void assemble_network_layer_packet();

    // Transport Layer
void init_transport_layer();
void read_transport_layer_packet();
void assemble_transport_layer_packet();

    // Application Layer
void init_application_layer();
void read_application_layer_packet();
void assemble_application_layer_packet();
void run_application();

// ENUMS
enum packet_indexes{
    // Physical Layer
    UPLINK_RSSI     = 0,
    UPLINK_QI       = 1,
    DOWNLINK_RSSI   = 2,
    DOWNLINK_QI     = 3,

    // MAC Layer
    MAC_COUNTER_MSB = 4, 
    MAC_COUNTER_LSB = 5,
    MAC3            = 6,
    MAC4            = 7,

    // Network Layer
    RECEIVER_ID     = 8,
    NET2            = 9,
    TRANSMITTER_ID  = 10,
    NET4            = 11,

    // Transport Layer
    DL_COUNTER_MSB = 12,
    DL_COUNTER_LSB = 13,
    UL_COUNTER_MSB = 14,
    UL_COUNTER_LSB = 15,

    // Application Layer
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
    PUMP_SIGNAL             = 27,
    LIGHT_SIGNAL            = 28,
    PUMP_INTERVAL_BYTE_0    = 29,
    PUMP_INTERVAL_BYTE_1    = 30,
    PUMP_DURATION_BYTE_0    = 31,
    PUMP_DURATION_BYTE_1    = 32,
    AUTOMATIC_MODE_TYPE = 33,
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
};

#endif