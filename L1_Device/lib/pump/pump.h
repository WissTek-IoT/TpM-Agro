#ifndef __PUMP_H__
#define __PUMP_H__

#include "utils.h"
#include "pinout.h"

// FUNCTIONS
void init_pump_system();
bool get_pump_state();
void turn_pump_on();
void turn_pump_off();
void control_pump_by_button();
void control_pump_by_signal(uint8_t signal);
void validate_pump_system();

#endif