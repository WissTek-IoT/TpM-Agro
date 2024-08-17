#include "mot.h"

byte dl_packet[PACKET_BYTES];
byte ul_packet[PACKET_BYTES];

uint32_t dl_packet_counter;
uint32_t ul_packet_counter;

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

void init_mot_protocol() {
    init_physical_layer();
    init_mac_layer();
    init_network_layer();
    init_transport_layer();
    init_application_layer();
}

void receive_mot_packet() {
    // If a packet was received, read it
    if (Serial.available() >= PACKET_BYTES) {
        // Reads the entire packet
        Serial.readBytes(dl_packet, PACKET_BYTES);
        clear_ul_packet();

        read_physical_layer_packet();
    }
}

void assemble_mot_packet() {
    // It is only needed to call application because each layer calls the one above it
    assemble_application_layer_packet();
}

void send_mot_packet() {
    // If it is possible to write all bytes from packet, send packet
    if (Serial.availableForWrite() >= PACKET_BYTES) {
        assemble_mot_packet();

        // Sends packet and waits for the transmission to complete
        Serial.write(ul_packet, PACKET_BYTES);
        Serial.flush();
        Serial.write("\n");
    }
}