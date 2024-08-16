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
  // validate_light_system();
  pinMode(AUTOMATIC_MODE_LED_PIN, OUTPUT);
  pinMode(AUTOMATIC_MODE_BUTTON_PIN, INPUT);
}

void loop() {
  // validate_WTK_TH();
  // turn_light_off();
  // validate_light_system();
  Serial.print(digitalRead(PUMP_BUTTON_PIN));
  Serial.print(" | ");
  Serial.print(digitalRead(LIGHT_BUTTON_PIN));
  Serial.print(" | ");
  Serial.println(digitalRead(AUTOMATIC_MODE_BUTTON_PIN));
  // Serial.print(" | ");
  digitalWrite(AUTOMATIC_MODE_LED_PIN, HIGH);
}