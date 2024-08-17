#include "mot.h"

void init_physical_layer() {
    Serial.begin(SERIAL_BAUD_RATE);
    init_pump_system();
    init_light_system();
    init_WTK_TH();
    init_control();
}

void read_physical_layer_packet() {

}

void assemble_physical_layer_packet() {

}