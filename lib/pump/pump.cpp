#include "pump.h"

uint16_t counter_button_pressed;
uint16_t counter_button_released;

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

void control_pump() {
    bool pump_button = check_if_pump_button_is_pressed();
    if (pump_button == true) {
        turn_pump_on();
        return;
    } 

    if (pump_button == false) {
        turn_pump_off();
        return;
    }
}

void validate_pump_system() {
    turn_pump_on();
    delay(3000);
    turn_pump_off();
}