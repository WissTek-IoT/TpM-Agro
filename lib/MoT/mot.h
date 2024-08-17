#ifndef __MOT_H__
#define __MOT_H__

#include "utils.h"
#include "pump.h"
#include "light.h"
#include "WTK_TH.h"
#include "mode.h"

// MACROS
    #define SERIAL_BAUD_RATE    9600

// FUNCTIONS
    // MoT
void init_mot_protocol();

    // Physical Layer
void init_physical_layer();

    // MAC Layer
void init_mac_layer();

    // Network Layer
void init_network_layer();

    // Transport Layer
void init_transport_layer();

    // Application Layer
void init_application_layer();


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
    APP1 = 16,
    APP2 = 17,
    APP3 = 18,
    APP4 = 19,
    APP5 = 20,
    APP6 = 21,
    APP7 = 22,
    APP8 = 23,
    APP9 = 24,
    APP10 = 25,
    APP11 = 26,
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
};


#endif