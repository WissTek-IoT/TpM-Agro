#include "utils.h"
#include "pump.h"
#include "light.h"
#include "WTK_TH.h"

void setup() {
  Serial.begin(9600);
  init_pump_system();
  init_light_system();
  init_WTK_TH();

  validate_pump_system();
  validate_light_system();
}

void loop() {
  validate_WTK_TH();
  turn_light_off();
}