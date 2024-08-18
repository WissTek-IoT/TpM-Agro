#include "mot.h"

void init_physical_layer() {
    Serial.begin(SERIAL_BAUD_RATE);
}

void read_physical_layer_packet() {
    read_mac_layer_packet();
}

void assemble_physical_layer_packet() {

}