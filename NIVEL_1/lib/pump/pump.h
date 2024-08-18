#ifndef __PUMP_H__
#define __PUMP_H__

#include "utils.h"
#include "pinout.h"

// MACROS

// FUNCTIONS
void init_pump_system();
void turn_pump_on();
void turn_pump_off();
void control_pump();
void validate_pump_system();

#endif