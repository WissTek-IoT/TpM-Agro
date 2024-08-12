#include "utils.h"
#include "DHT22.h"
#include "WTK_TH.h"

void setup() {
  Serial.begin(9600);
  init_DHT22_sensors();
  init_WTK_TH();

  pinMode(RELAY_PIN,OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);
  delay(3000);
  digitalWrite(RELAY_PIN,LOW);
}

void loop() {
  validate_WTK_TH();
}