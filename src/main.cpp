#include "mot.h"

unsigned long last_time = 0;
unsigned long current_time = 0;

void setup() {
  init_mot_protocol();
}

void loop() {
  current_time = millis();
  run_application();

  if (current_time - last_time >= 3000) {
    send_mot_packet();
    last_time = current_time;
  }
}