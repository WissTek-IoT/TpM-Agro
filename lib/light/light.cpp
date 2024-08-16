void init_light_system();
void turn_light_on();
void turn_light_off();
void read_ambient_light();
void validate_light();

#include "light.h"

void init_light_system() {
    pinMode(LIGHT_RELAY_PIN, OUTPUT);
    turn_light_off();
}

void turn_light_on() {
    digitalWrite(LIGHT_RELAY_PIN, HIGH);
}

void turn_light_off() {
    digitalWrite(LIGHT_RELAY_PIN, LOW);
}

void validate_light_system() {
    turn_light_on();
    delay(4000);
    turn_light_off();
}