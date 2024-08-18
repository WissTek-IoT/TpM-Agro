#include "mot.h"

void init_transport_layer() {
    // Clears UpLink and DownLink packets
    ul_packet_counter = 0;
    dl_packet_counter = 0;
}

void read_transport_layer_packet() {
    // A successful packet was received
    dl_packet_counter++;
    read_application_layer_packet();
}

void assemble_transport_layer_packet() {
    ul_packet_counter++;

    ul_packet[UL_COUNTER_LSB] = ul_packet_counter%256;
    ul_packet[UL_COUNTER_MSB] = ul_packet_counter/256;

    ul_packet[DL_COUNTER_LSB] = dl_packet_counter%256;
    ul_packet[DL_COUNTER_MSB] = dl_packet_counter/256;

    assemble_network_layer_packet();
}