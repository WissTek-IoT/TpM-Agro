#ifndef __MODE_H__
#define __MODE_H__

#include "utils.h"
#include "pinout.h"

// ENUMS
enum controller_type{
    MANUAL_CONTROL,
    AUTOMATIC_CONTROL
};

// MACROS

// FUNCTIONS
void init_control();
controller_type control_mode();

#endif