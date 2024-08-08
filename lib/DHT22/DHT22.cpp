#include "DHT22.h"

DHT_Unified DHT22_SENSOR_1(DHT22_PIN, DHT_TYPE);

void init_DHT22_sensors() {
  // Initialize device.
  DHT22_SENSOR_1.begin();
}

void validate_DHT22_sensors() {
    sensors_event_t event;

    DHT22_SENSOR_1.temperature().getEvent(&event);
    if (isnan(event.temperature)) {
        Serial.println("Error reading temperature!");
        return;
    }
    Serial.print("Temperature: ");
    Serial.print(event.temperature);
    Serial.println(" °C");

    DHT22_SENSOR_1.humidity().getEvent(&event);
    if (isnan(event.relative_humidity)) {
        Serial.println("Error reading humidity!");
        return;
    }
    Serial.print("Humidity: ");
    Serial.print(event.relative_humidity);
    Serial.println("%");

    Serial.println("---------------------------------");
    delay(READING_INTERVAL_MS);
}