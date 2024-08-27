#include "mot.h"

void setup() {
  init_mot_protocol();
}

void loop() {
  run_application();
  receive_mot_packet();
}