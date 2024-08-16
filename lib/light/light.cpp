void init_light_system();
void turn_light_on();
void turn_light_off();
void read_ambient_light();
void validate_light();

#include "light.h"

SI114X light_sensor = SI114X();

void init_light_system() {
    pinMode(LIGHT_RELAY_PIN, OUTPUT);
    turn_light_off();

    while (!light_sensor.Begin()) {
        Serial.println("Configurando o Sensor de Luz");
    }
    // Wire1.begin(A4, A5);
    // Wire.begin();
    Serial.println("Sistema de iluminação inicializado.");
}

void turn_light_on() {
    digitalWrite(LIGHT_RELAY_PIN, HIGH);
}

void turn_light_off() {
    digitalWrite(LIGHT_RELAY_PIN, LOW);
}

uint16_t read_visible_light() {
    return light_sensor.ReadVisible();
}

uint16_t read_IR_light() {
    return light_sensor.ReadIR();
}

float read_UV_index() {
    return (float)light_sensor.ReadUV()/100;
}

void validate_light_system() {
    // turn_light_on();
    // delay(4000);
    // turn_light_off();

    Serial.print("//--------------------------------------//\r\n");
    Serial.print("Vis: "); Serial.println(read_visible_light());
    Serial.print("IR: "); Serial.println(read_IR_light());
    //the real UV value must be div 100 from the reg value , datasheet for more information.
    Serial.print("UV: ");  Serial.println(read_UV_index());
    delay(1000);
}