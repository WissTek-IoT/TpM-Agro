#include "mot.h"

unsigned long last_time = 0;
unsigned long current_time = 0;

unsigned long long begin = 0;
unsigned long long end = 0;

void setup() {
  init_mot_protocol();
}

void loop() {
  current_time = millis();

  run_application();
  receive_mot_packet();
  // if (current_time - last_time >= 1000) {
  //   // begin = micros();
  //   send_mot_packet();
  //   // Serial.println((micros()-begin)/1000.0);
  //   last_time = current_time;
  // }
}