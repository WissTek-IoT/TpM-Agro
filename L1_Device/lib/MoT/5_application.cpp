#include "mot.h"

// Reading Variables
float current_temperature;
float current_humidity;

uint16_t current_visible_light_intensity;
uint16_t current_ir_light_intensity;
float current_uv_index;

controller_type control;

// Control Variables
uint8_t pump_signal     = 0;
uint8_t light_signal    = 0;
unsigned long pump_activation_interval = 1800000;
unsigned long pump_activation_duration = 30000;
uint8_t automatic_mode_type       = 0;

unsigned long current_time = 0;
unsigned long last_time = 0;

bool timer_enabled = false;
unsigned long timer_begin = 0;
unsigned long finish_time = 0;

bool pump_enabled = 0;
bool light_enabled = 0;

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

    // Assemble Actuators data
    ul_packet[IS_PUMP_ENABLED]  = pump_enabled;
    ul_packet[IS_LIGHT_ENABLED] = light_enabled;

    assemble_transport_layer_packet();
}

void read_application_layer_packet() {
    pump_signal     = dl_packet[PUMP_SIGNAL];
    light_signal    = dl_packet[LIGHT_SIGNAL];

    pump_activation_duration = (256*(unsigned long)dl_packet[PUMP_DURATION_BYTE_1] + (unsigned long)dl_packet[PUMP_DURATION_BYTE_0]) * 1000; // Assembles the data and converts into milliseconds
    pump_activation_interval = (256*(unsigned long)dl_packet[PUMP_INTERVAL_BYTE_1] + (unsigned long)dl_packet[PUMP_INTERVAL_BYTE_0]) * 1000; // Same as above

    automatic_mode_type = dl_packet[AUTOMATIC_MODE_TYPE];
}

/*  Periodic Automatic Control implies that the nutrient pump will be activated at a given interval for a given duration.
    However, light control is still done by python as it depends on having a RTC.
*/
void run_periodic_automatic_control() {
    current_time = millis();
    if ((current_time - last_time >= pump_activation_interval) && timer_enabled == false) {
        timer_enabled = true;
        timer_begin = millis();
    }

    if (timer_enabled == true) {
        if (current_time - timer_begin <= pump_activation_duration) {
            pump_enabled = HIGH;
            turn_pump_on();
        } else {
            pump_enabled = LOW;
            turn_pump_off();
            timer_enabled = false;
            last_time = millis();
        }
    }
    light_enabled = light_signal;
    control_light_by_signal(light_signal);
}

// Receives ML signal from packet and applies it to given devices.
void run_ml_automatic_control() {
    control_pump_by_signal(pump_signal);
    control_light_by_signal(light_signal);

    light_enabled = light_signal;
    pump_enabled = pump_signal;
}

// Runs either Periodic or Machine Learning Automatic Control
void run_automatic_control() {
    switch (automatic_mode_type) {
        case AUTOMATIC_PERIODIC_MODE:
            run_periodic_automatic_control();
            break;
        case AUTOMATIC_ML_MODE:
            run_ml_automatic_control();
            break;
        default:
            break;
    }
}

// Control given devices by external buttons
void run_manual_control() {
    control_pump_by_button();
    control_light_by_button();

    pump_enabled    = get_pump_state();
    light_enabled   = get_light_state();
}

void run_application() {
    // Temperature and Humidity
    current_temperature             = get_temperature(TH_AVERAGE);
    current_humidity                = get_relative_humidity(TH_AVERAGE);

    // Light Intensity
    current_visible_light_intensity = read_visible_light();
    current_ir_light_intensity      = read_IR_light();
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
            run_automatic_control();
            break;
        case MANUAL_CONTROL:
            run_manual_control();
            break;
        default:
            break;
    }
}