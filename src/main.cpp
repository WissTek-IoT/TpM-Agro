#include "utils.h"
#include "pump.h"
#include "light.h"
#include "WTK_TH.h"
#include "mode.h"

void setup() {
  Serial.begin(9600);
  init_pump_system();
  init_light_system();
  init_WTK_TH();
  init_control();

  // validate_pump_system();
  // validate_light_system();
}

void loop() {
  switch (control_mode()) {
    case AUTOMATIC_CONTROL:
      break;
    case MANUAL_CONTROL:
      control_pump();
      control_light();
      break;

    default:
      break;
  }
}