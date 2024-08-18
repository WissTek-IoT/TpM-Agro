#include "pump.h"

uint16_t counter_button_pressed;
uint16_t counter_button_released;

bool pump_button = false;

void init_pump_system() {
    pinMode(PUMP_RELAY_PIN, OUTPUT);
    pinMode(PUMP_BUTTON_PIN, INPUT);
    turn_pump_off();
}

void turn_pump_on() {
    digitalWrite(PUMP_RELAY_PIN, HIGH);
}

void turn_pump_off() {
    digitalWrite(PUMP_RELAY_PIN, LOW);
}

bool check_if_pump_button_is_pressed() {
    // While the pump button is being pressed, it has to activate the nutrient pump
    if (digitalRead(PUMP_BUTTON_PIN) == HIGH) return true;
    return false;
}

void control_pump_by_button() {
    pump_button = check_if_pump_button_is_pressed();
    if (pump_button == true) turn_pump_on();
    else turn_pump_off();
}

void control_pump_by_signal(uint8_t signal) {
    digitalWrite(PUMP_RELAY_PIN, signal);
}

bool get_pump_state() {
    return pump_button;
}

void validate_pump_system() {
    turn_pump_on();
    delay(3000);
    turn_pump_off();
}