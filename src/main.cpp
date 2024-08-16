#include "utils.h"
#include "pump.h"
#include "WTK_TH.h"

void setup() {
  Serial.begin(9600);
  init_pump();
  init_WTK_TH();

  validate_pump();
}

void loop() {
  validate_WTK_TH();
}