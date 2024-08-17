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

}