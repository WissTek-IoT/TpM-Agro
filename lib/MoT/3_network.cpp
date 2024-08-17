#include "mot.h"

void init_network_layer() {

}

void read_network_layer_packet() {
    if (dl_packet[RECEIVER_ID] == MY_ID) read_transport_layer_packet();
}

void assemble_network_layer_packet() {

}