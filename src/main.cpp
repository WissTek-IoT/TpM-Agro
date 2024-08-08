#include "utils.h"
#include "DHT22.h"

void setup() {
  Serial.begin(9600);
  init_DHT22_sensors();
}

void loop() {
  validate_DHT22_sensors();
}