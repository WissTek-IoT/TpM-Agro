#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "utils.h"
#include "pinout.h"

// MACROS

// FUNCTIONS
void init_light_system();

    // ACTUATOR FUNCTIONS
void turn_light_on();
void turn_light_off();

    // SENSOR FUNCTIONS
void read_ambient_light();

    // VALIDATION FUNCTIONS
void validate_light_system();

#endif