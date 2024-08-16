#include "mode.h"

unsigned long mode_counter = 0;
bool mode_state = false;
controller_type controller = AUTOMATIC_CONTROL;

void init_control() {
    pinMode(AUTOMATIC_MODE_LED_PIN, OUTPUT);
    pinMode(AUTOMATIC_MODE_BUTTON_PIN, INPUT);
}

bool check_if_mode_button_is_pressed() {
    // Pressing the button causes the light to switch from ON to OFF or vice-versa
    if (digitalRead(AUTOMATIC_MODE_BUTTON_PIN) == HIGH) {
        mode_counter++;
    } else {
        if (mode_counter >= BUTTON_COUNTER_THRESHOLD) {
            mode_state = !mode_state;
        }
        mode_counter = 0;
    }

    return mode_state;
}

controller_type control_mode() {
    bool mode_button = check_if_mode_button_is_pressed();
    if (mode_button == true) {
        controller = AUTOMATIC_CONTROL;
        digitalWrite(AUTOMATIC_MODE_LED_PIN, HIGH);
    }
    else {
        controller = MANUAL_CONTROL;
        digitalWrite(AUTOMATIC_MODE_LED_PIN, LOW);
    }

    return controller;
}