#ifndef __WTK_TH_H__
#define __WTK_TH_H__

#include "utils.h"
#include "pinout.h"

// ENUMS
enum th_sensor_index{
    TH_SENSOR_1,
    TH_SENSOR_2,
    TH_AVERAGE
};

// MACROS
#define WTK_TH_INPUT_VOLTAGE                            5.0
#define NUMBER_OF_WTK_TH_SENSORS                        2
#define WTK_TEMPERATURE_READING_SAMPLE_TIME             5
#define WTK_TEMPERATURE_NUMBER_OF_SAMPLES_PER_READING   100

// FUNCTIONS
void init_WTK_TH();
float get_temperature(th_sensor_index sensor_index);
float get_relative_humidity(th_sensor_index sensor_index);
void validate_WTK_TH();

#endif