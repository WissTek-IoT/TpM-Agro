#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "utils.h"
#include "pinout.h"
#include <Wire.h>
#include <SI114X.h>

// MACROS

// FUNCTIONS
void init_light_system();

    // ACTUATOR FUNCTIONS
void turn_light_on();
void turn_light_off();
void control_light_by_button();
void control_light_by_signal(uint8_t signal);

    // SENSOR FUNCTIONS
uint16_t read_visible_light();
uint16_t read_IR_light();
float read_UV_index();

    // VALIDATION FUNCTIONS
void validate_light_system();

#endif