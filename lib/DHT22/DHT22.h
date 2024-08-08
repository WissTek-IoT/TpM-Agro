#ifndef __DHT22_H__
#define __DHT22_H__

#include "utils.h"
#include "pinout.h"
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>

// MACROS
#define DHT22_PIN   5
#define DHT_TYPE    DHT22

#define READING_INTERVAL_MS 1000

// FUNCTIONS
void init_DHT22_sensors();
void validate_DHT22_sensors();

#endif