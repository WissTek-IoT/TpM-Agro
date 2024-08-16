#include "pump.h"

void init_pump_system() {
    pinMode(PUMP_RELAY_PIN, OUTPUT);
    turn_pump_off();
}

void turn_pump_on() {
    digitalWrite(PUMP_RELAY_PIN, HIGH);
}

void turn_pump_off() {
    digitalWrite(PUMP_RELAY_PIN, LOW);
}

void validate_pump_system() {
    turn_pump_on();
    delay(3000);
    turn_pump_off();
}