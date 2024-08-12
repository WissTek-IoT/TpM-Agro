#include "WTK_TH.h"

uint8_t WTK_T_PINS[NUMBER_OF_WTK_TH_SENSORS] = {WKT_TEMPERATURE_SENSOR_1_PIN, WKT_TEMPERATURE_SENSOR_2_PIN};
uint8_t WTK_H_PINS[NUMBER_OF_WTK_TH_SENSORS] = {WKT_HUMIDITY_SENSOR_1_PIN, WKT_HUMIDITY_SENSOR_2_PIN};
float humidity[NUMBER_OF_WTK_TH_SENSORS + 1]; // Extra index to include average value
float temperature[NUMBER_OF_WTK_TH_SENSORS + 1]; // Extra index to include average value

void init_WTK_TH() {
  
}

void read_temperature() {
  float average_temperature[NUMBER_OF_WTK_TH_SENSORS];

  // Calculate temperature for each sensor
  for (uint8_t current_sensor = 0; current_sensor < NUMBER_OF_WTK_TH_SENSORS; current_sensor++) {
    for (uint8_t i = 0; i < WTK_TEMPERATURE_NUMBER_OF_SAMPLES_PER_READING; i++) {
      // For more information on that equation, read temperature sensors' datasheet
      average_temperature[current_sensor] += ((((analogRead(WTK_T_PINS[current_sensor])/1023.0) * WTK_TH_INPUT_VOLTAGE) - 0.5)/0.01);
      delay(WTK_TEMPERATURE_READING_SAMPLE_TIME);
    } 
    average_temperature[current_sensor] /= WTK_TEMPERATURE_NUMBER_OF_SAMPLES_PER_READING;

    // The below equation is due to a second calibration made comparing the sensor readings with an external temperature/humidity sensor
    average_temperature[current_sensor] = 1.5769*average_temperature[current_sensor] - 8.384;

    // Cleans the second decimal digit
    average_temperature[current_sensor] = round(average_temperature[current_sensor]*10)/10.0;

    temperature[current_sensor] = average_temperature[current_sensor];
  }

  // Calculate the average temperature based on all sensor readings
  for (uint8_t current_sensor = 0; current_sensor < NUMBER_OF_WTK_TH_SENSORS; current_sensor++) {
    average_temperature[NUMBER_OF_WTK_TH_SENSORS] += average_temperature[current_sensor];
  }
  average_temperature[NUMBER_OF_WTK_TH_SENSORS] /= NUMBER_OF_WTK_TH_SENSORS;
  temperature[NUMBER_OF_WTK_TH_SENSORS] = average_temperature[NUMBER_OF_WTK_TH_SENSORS];
}

void read_relative_humidity() {
  float average_humidity = 0;

  // Calculate Relative Humidity for each sensor
  for (uint8_t current_sensor = 0; current_sensor < NUMBER_OF_WTK_TH_SENSORS; current_sensor++) {
    // For more information on that, see HIH-4030's (humidity sensor) datasheet 
    float current_humidity = ((analogRead(WTK_H_PINS[current_sensor])/1023.0) - 0.16)/0.0062;
    current_humidity = current_humidity/(1.0546 - 0.00216*temperature[current_sensor]);

    humidity[current_sensor] = current_humidity;
  }

  // Calculate the average Relative Humidity based on all sensor readings
  for (uint8_t current_sensor = 0; current_sensor < NUMBER_OF_WTK_TH_SENSORS; current_sensor++) {
    average_humidity += humidity[current_sensor];
  }
  average_humidity /= NUMBER_OF_WTK_TH_SENSORS;
  humidity[NUMBER_OF_WTK_TH_SENSORS] = average_humidity;
}

float get_temperature(th_sensor_index sensor_index) {
  read_temperature();
  return temperature[sensor_index];
}

float get_relative_humidity(th_sensor_index sensor_index) {
  read_relative_humidity();
  return humidity[sensor_index];
}

void validate_WTK_TH() {
  read_temperature();
  read_relative_humidity();

  Serial.print("WTK");
  for (uint8_t current_sensor = 0; current_sensor < NUMBER_OF_WTK_TH_SENSORS; current_sensor++) {
    Serial.print("| Rel. Humid.[");
    Serial.print(current_sensor);
    Serial.print("]: ");
    Serial.print(humidity[current_sensor]);
    Serial.print("%   ");
  }

  Serial.println();

  Serial.print("WTK");
  for (uint8_t current_sensor = 0; current_sensor < NUMBER_OF_WTK_TH_SENSORS; current_sensor++) {
    Serial.print("| Temperature[");
    Serial.print(current_sensor);
    Serial.print("]: ");
    Serial.print(temperature[current_sensor]);
    Serial.print(" °C ");
  }
  
  Serial.println();

  Serial.print("WTK| Avg. RH: ");
  Serial.print(humidity[NUMBER_OF_WTK_TH_SENSORS]);
  Serial.print("% | Avg. Temp: ");
  Serial.print(temperature[NUMBER_OF_WTK_TH_SENSORS]);
  Serial.println(" °C");
  Serial.println();
  delay(500);
}