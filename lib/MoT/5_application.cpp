#include "mot.h"

float current_temperature;
float current_humidity;

uint16_t current_visible_light_intensity;
uint16_t current_ir_light_intensity;
float current_uv_index;

controller_type control;

void init_application_layer() {
    init_pump_system();
    init_light_system();
    init_WTK_TH();
    init_control();
}

void assemble_application_layer_packet() {
    // Assemble temperature data
    ul_packet[TEMPERATURE_BYTE_0] = (uint16_t)(current_temperature*10.0)%256;
    ul_packet[TEMPERATURE_BYTE_1] = (uint16_t)(current_temperature*10.0)/256;

    // Assemble humidity data
    ul_packet[HUMIDITY_BYTE_0] = (uint16_t)(current_humidity*10.0)%256;
    ul_packet[HUMIDITY_BYTE_1] = (uint16_t)(current_humidity*10.0)/256;

    // Assemble visible light intensity data
    ul_packet[VISIBLE_LIGHT_BYTE_0] = (uint16_t)(current_visible_light_intensity)%256;
    ul_packet[VISIBLE_LIGHT_BYTE_1] = (uint16_t)(current_visible_light_intensity)/256;
    // Assemble IR light intensity data
    ul_packet[IR_LIGHT_BYTE_0] = (uint16_t)(current_ir_light_intensity)%256;
    ul_packet[IR_LIGHT_BYTE_1] = (uint16_t)(current_ir_light_intensity)/256;
    // Assemble UV index data
    ul_packet[UV_INDEX_BYTE_0] = (uint16_t)(current_uv_index*10.0)%256;
    ul_packet[UV_INDEX_BYTE_1] = (uint16_t)(current_uv_index*10.0)/256;

    // Assemble control mode data
    ul_packet[CONTROL_TYPE_INDEX] = control;

    assemble_transport_layer_packet();
}

void read_application_layer_packet() {

}

void run_application() {
    // Temperature and Humidity
    current_temperature             = get_temperature(TH_AVERAGE);
    current_humidity                = get_relative_humidity(TH_AVERAGE);

    // Light Intensity
    current_visible_light_intensity = read_visible_light();
    current_ir_light_intensity      = read_visible_light();
    current_uv_index                = read_UV_index();

    // constrain all variables so they don't cause packet errors
    current_temperature             = constrain(current_temperature,                0, UINT16_MAX);
    current_humidity                = constrain(current_humidity,                   0, UINT16_MAX);
    current_visible_light_intensity = constrain(current_visible_light_intensity,    0, UINT16_MAX);
    current_ir_light_intensity      = constrain(current_ir_light_intensity,         0, UINT16_MAX);
    current_uv_index                = constrain(current_uv_index,                   0, 20);

    control = control_mode();
    switch (control) {
    case AUTOMATIC_CONTROL:
        break;
    case MANUAL_CONTROL:
        control_pump();
        control_light();
        break;

    default:
        break;
    }
}