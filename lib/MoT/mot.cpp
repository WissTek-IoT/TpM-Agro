#include "mot.h"

byte dl_packet[PACKET_BYTES];
byte ul_packet[PACKET_BYTES];

void init_mot_protocol() {
    init_physical_layer();
    init_mac_layer();
    init_network_layer();
    init_transport_layer();
    init_application_layer();
}

void receive_mot_packet() {
    if (Serial.available() >= PACKET_BYTES) {
        Serial.readBytes(dl_packet, PACKET_BYTES);
        clear_ul_packet();
        
        read_physical_layer_packet();
    }
}

void send_mot_packet() {

}

void clear_ul_packet() {
    for (uint16_t i = 0; i < PACKET_BYTES; i++) {
        ul_packet[i] = 0;
    }
}

void clear_dl_packet() {
    for (uint16_t i = 0; i < PACKET_BYTES; i++) {
        dl_packet[i] = 0;
    }
}