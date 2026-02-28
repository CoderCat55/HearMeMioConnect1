#include <AsyncTCP.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <ESPAsyncWebServer.h>
#include <Wire.h>
#include <math.h>
#include <cfloat>
#include "Arduino.h"

// WiFi credentials
const char* ssid = "duybeni";
const char* password = "123456789";

// Pin definitions
#define MPU_SCL 22
#define MPU_SDA 21
#define EMG2_PIN 32

float threshold = 6.0;

// MPU6050 address
const int MPU_ADDR = 0x68;

unsigned long lastConnectionCleanup = 0;

// EMG processing variables
const int EMG_WINDOW_SIZE = 30;
int emg1Values[EMG_WINDOW_SIZE] = {0};
int emg1Index = 0;
float emg1Avg = 0;


struct SensorData {
  // Accelerometer data
  float ax1 = 0, ay1 = 0, az1 = 0;  // From ESP-E (client)
  float ax2 = 0, ay2 = 0, az2 = 0;  // From ESP-D (server)

  // All EMG values
  float emg1 = 0;  // From ESP-E (D32)
  float emg2 = 0;  // From ESP-D (D32)

  //system control values
  int cw =0;
  int wc=0;

  
  // System state
  int connections = 0;
  int calword = 0;
  int cr = 0;
};

SensorData sensorData;
// Calibration data arrays
const int MAX_SAMPLES = 20;
const int NUM_WORDS = 10;
int calibrationIndex[NUM_WORDS] = {0}; // Index for each word's calibration data

const int NUM_FEATURES = 8;  // 6 accel + 2 EMG
float calibrationData[NUM_WORDS][MAX_SAMPLES][NUM_FEATURES];

AsyncWebServer server(80); // server port 80
// Initialize MPU6050
void initMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // Set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
}

// Server-side - EMG RMS Filter (for muscle activation detection)
float processEMG(float emgSmoothed) {
  static float rmsBuffer[20] = {0};
  static int index = 0;
  static float sumSquares = 0;

  sumSquares -= rmsBuffer[index] * rmsBuffer[index];
  rmsBuffer[index] = emgSmoothed;
  sumSquares += emgSmoothed * emgSmoothed;
  index = (index + 1) % 20;

  return sqrt(sumSquares / 20); // RMS value
}

int classifyGesture() {
  // Processed sensor data (8 features: 2 EMG + 6 Accel)
  float currentData[8] = {
    processEMG(sensorData.emg1), processEMG(sensorData.emg2), // EMG
    sensorData.ax1, sensorData.ay1, sensorData.az1,           // Accel 1
    sensorData.ax2, sensorData.ay2, sensorData.az2            // Accel 2
  };

  // Feature normalization/scaling factors (adjust based on your data ranges)
  float featureScales[8] = {
    1.0, 1.0,     // EMG scaling
    10.0, 10.0, 10.0,  // Accel 1 scaling (accelerometer typically has smaller values)
    10.0, 10.0, 10.0   // Accel 2 scaling
  };

  // Structure to store distances and corresponding gestures
  struct DistanceGesture {
    float distance;
    int gesture;
  };

  // Collect all valid samples from all gestures
  const int MAX_TOTAL_SAMPLES = NUM_WORDS * MAX_SAMPLES;
  DistanceGesture allSamples[MAX_TOTAL_SAMPLES];
  int totalSamples = 0;

  // Calculate distances to all calibrated samples
  for (int gesture = 1; gesture <= NUM_WORDS; gesture++) {
    int calIdx = gesture - 1;
    if (calibrationIndex[calIdx] == 0) continue;

    for (int sample = 0; sample < calibrationIndex[calIdx]; sample++) {
      float distance = 0;

      // Calculate weighted Euclidean distance with proper feature scaling
      for (int i = 0; i < 8; i++) {
        float diff = (currentData[i] - calibrationData[calIdx][sample][i]) * featureScales[i];
        
        // Apply feature-specific weights
        float weight = (i < 2) ? 1.0 : 2.0;  // EMG vs Accelerometer weights
        distance += weight * diff * diff;
      }

      distance = sqrt(distance);

      // Store all valid samples (within reasonable threshold)
      if (distance < 15.0 && totalSamples < MAX_TOTAL_SAMPLES) {  // Expanded threshold for collection
        allSamples[totalSamples].distance = distance;
        allSamples[totalSamples].gesture = gesture;
        totalSamples++;
      }
    }
  }

  if (totalSamples == 0) return 0;  // No valid samples found

  // Sort samples by distance (simple bubble sort - efficient for small datasets)
  for (int i = 0; i < totalSamples - 1; i++) {
    for (int j = 0; j < totalSamples - i - 1; j++) {
      if (allSamples[j].distance > allSamples[j + 1].distance) {
        DistanceGesture temp = allSamples[j];
        allSamples[j] = allSamples[j + 1];
        allSamples[j + 1] = temp;
      }
    }
  }

  // Implement adaptive K selection
  int k = min(7, totalSamples);  // Use up to 7 nearest neighbors, but not more than available samples
  if (totalSamples >= 15) k = 7;
  else if (totalSamples >= 9) k = 5;
  else if (totalSamples >= 5) k = 3;
  else k = min(3, totalSamples);

  // Vote among K nearest neighbors with distance-weighted voting
  float gestureScores[NUM_WORDS + 1] = {0};  // Index 0 unused, 1-NUM_WORDS for gestures
  float totalWeight = 0;

  for (int i = 0; i < k; i++) {
    int gesture = allSamples[i].gesture;
    float distance = allSamples[i].distance;
    
    // Distance-weighted voting (closer samples have more influence)
    // Using inverse distance weighting with smoothing
    float weight = 1.0 / (1.0 + distance);
    gestureScores[gesture] += weight;
    totalWeight += weight;
  }

  // Find gesture with highest weighted score
  int bestGesture = 0;
  float maxScore = 0;
  float secondMaxScore = 0;

  for (int gesture = 1; gesture <= NUM_WORDS; gesture++) {
    if (gestureScores[gesture] > maxScore) {
      secondMaxScore = maxScore;
      maxScore = gestureScores[gesture];
      bestGesture = gesture;
    } else if (gestureScores[gesture] > secondMaxScore) {
      secondMaxScore = gestureScores[gesture];
    }
  }

  // Confidence check - ensure clear winner
  float confidence = maxScore / totalWeight;
  float margin = (maxScore - secondMaxScore) / totalWeight;

  // Final validation: distance check on best match
  float avgDistanceToWinner = 0;
  int winnerCount = 0;
  
  for (int i = 0; i < k; i++) {
    if (allSamples[i].gesture == bestGesture) {
      avgDistanceToWinner += allSamples[i].distance;
      winnerCount++;
    }
  }
  
  if (winnerCount > 0) {
    avgDistanceToWinner /= winnerCount;
  }

  // Return result based on confidence, margin, and average distance
  if (confidence > 0.3 && margin > 0.1 && avgDistanceToWinner < 8.0) {
    return bestGesture;
  }
  
  return 0;  // Not confident enough in classification
}

String getSensorDataJSON() {
DynamicJsonDocument doc(1024);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);  // request a total of 6 registers

  // Accelerometer data
  doc["ax1"] = sensorData.ax1;
  doc["ay1"] = sensorData.ay1;
  doc["az1"] = sensorData.az1;
  doc["ax2"] = sensorData.ax2 =(Wire.read() << 8 | Wire.read()) / 16384.0;  // X-axis value ;
  doc["ay2"] = sensorData.ay2 =(Wire.read() << 8 | Wire.read()) / 16384.0;  // Y-axis value;
  doc["az2"] = sensorData.az2 = (Wire.read() << 8 | Wire.read()) / 16384.0;  // Z-axis value;
  
  // EMG data
  
  doc["emg1"] = sensorData.emg1;
  doc["emg2"] = sensorData.emg2 = readEMG(EMG2_PIN);
 
  
  // System state
  doc["cr"] = sensorData.cr;
  doc["calword"] = sensorData.calword;
  doc["connections"] = sensorData.connections;
  String json;
  serializeJson(doc, json);
  return json;
}


float readEMG(int pin) {
    int rawValue = analogRead(pin);
    emg1Avg = emg1Avg - emg1Values[emg1Index]/float(EMG_WINDOW_SIZE) + rawValue/float(EMG_WINDOW_SIZE);
    emg1Values[emg1Index] = rawValue;
    emg1Index = (emg1Index + 1) % EMG_WINDOW_SIZE;
    return emg1Avg;
}

void clearCalibrationData(int wordIndex) {
  if (wordIndex < 0 || wordIndex >= NUM_WORDS) return;
  
  String prefix = String(wordIndex + 1) + "_";
  
  calibrationIndex[wordIndex] = 0; // Reset in-memory index
  
  Serial.printf("Cleared all data for word %d\n", wordIndex + 1);
}

void saveCalibrationSample(int wordIndex) {
  // Validate input
  if (wordIndex < 0 || wordIndex >= NUM_WORDS) {
    Serial.println("Error: Invalid word index");
    return;
  }

  // Shift samples if buffer is full (FIFO)
  if (calibrationIndex[wordIndex] >= MAX_SAMPLES) {
    for (int i = 0; i < MAX_SAMPLES-1; i++) {
      memcpy(calibrationData[wordIndex][i], calibrationData[wordIndex][i+1], NUM_FEATURES * sizeof(float));
    }
    calibrationIndex[wordIndex] = MAX_SAMPLES-1;
  }

  // Store current sensor data (8 features)
  calibrationData[wordIndex][calibrationIndex[wordIndex]][0] = processEMG(sensorData.emg1); // EMG1 (processed)
  calibrationData[wordIndex][calibrationIndex[wordIndex]][1] = processEMG(sensorData.emg2); // EMG2 (processed)
  calibrationData[wordIndex][calibrationIndex[wordIndex]][2] = sensorData.ax1; // Accel1 X
  calibrationData[wordIndex][calibrationIndex[wordIndex]][3] = sensorData.ay1; // Accel1 Y
  calibrationData[wordIndex][calibrationIndex[wordIndex]][4] = sensorData.az1; // Accel1 Z
  calibrationData[wordIndex][calibrationIndex[wordIndex]][5] = sensorData.ax2; // Accel2 X
  calibrationData[wordIndex][calibrationIndex[wordIndex]][6] = sensorData.ay2; // Accel2 Y
  calibrationData[wordIndex][calibrationIndex[wordIndex]][7] = sensorData.az2; // Accel2 Z


  // Update and save count
  calibrationIndex[wordIndex]++;
  Serial.printf("Saved sample %d for word %d\n", calibrationIndex[wordIndex], wordIndex + 1);
}

void notFound(AsyncWebServerRequest *request)
{
  request->send(404, "text/plain", "Page Not found");
}

void setup(void)
{
  
  Serial.begin(115200);
  
  WiFi.softAP(ssid, password);
  Serial.println("softap");
  Serial.println("");
  Serial.println(WiFi.softAPIP());

  // Add this at the beginning of your setup() to verify URLs
Serial.println("Lets starttt:");



  if (MDNS.begin("duybeni")) { //esp.local/
    Serial.println("MDNS responder started");
  }
  // Initialize hardware
Wire.begin(MPU_SDA, MPU_SCL);
initMPU6050();

Serial.println("System initialization complete");


// Serve your HTML page
server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
  sensorData.connections++;
  Serial.printf("New request from %s\n", request->client()->remoteIP().toString().c_str());
});

server.on("/classify", HTTP_GET, [](AsyncWebServerRequest *request){
    // Perform classification immediately
    sensorData.cr = classifyGesture();
    Serial.print("Classification requested - Result: ");
    Serial.println(sensorData.cr);
    request->send(200, "application/json", getSensorDataJSON()); // Use getSensorDataJSON() instead of json
});

// Endpoint to set calibration word
server.on("/setcw", HTTP_GET, [](AsyncWebServerRequest *request){
    if(request->hasParam("value")){
        int word = request->getParam("value")->value().toInt();
        if(word >= 1 && word <= NUM_WORDS) {
            saveCalibrationSample(word-1); // Convert to 0-based index
            Serial.print("Calibration sample saved for word: "); Serial.println(word);
            request->send(200, "text/plain", "Saved sample for word " + String(word));
        } else {
            request->send(400, "text/plain", "Invalid word index (1-" + String(NUM_WORDS) + ")");
        }
    } else {
        request->send(400, "text/plain", "Missing value parameter");
    }
});

// Endpoint to delete calibration word
server.on("/deletecw", HTTP_GET, [](AsyncWebServerRequest *request){
    if(request->hasParam("value")){
        int word = request->getParam("value")->value().toInt();
        if(word >= 1 && word <= NUM_WORDS) {
            clearCalibrationData(word-1);
            Serial.print("Calibration cleared for word: "); Serial.println(word);
            request->send(200, "text/plain", "Cleared calibration for word " + String(word));
        } else {
            request->send(400, "text/plain", "Invalid word index (1-" + String(NUM_WORDS) + ")");
        }
    } else {
        request->send(400, "text/plain", "Missing value parameter");
    }
});


// Endpoint for sensor data
server.on("/data", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(200, "application/json", getSensorDataJSON());
});

// Handle POST requests to /update endpoint
server.on(
    "/update", 
    HTTP_POST, 
    [](AsyncWebServerRequest *request) {
        // This handler will be called when the HTTP request completes
        // but we'll actually process the body in the onBody handler below
        request->send(200, "text/plain", "OK");
    },
    NULL,  // onUpload handler (not needed)
    [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
        // This is the onBody handler where we actually get the POST data
        // Create a static buffer for the JSON
        static char json[256];
        
        // Copy incoming data chunk to our buffer, making sure we don't overflow
        size_t copyLen = min(len, sizeof(json) - 1);
        memcpy(json, data, copyLen);
        json[copyLen] = 0;  // Null terminate
        
        // Parse JSON
        DynamicJsonDocument doc(256);
        DeserializationError error = deserializeJson(doc, json);
        
        if (!error) {
         // Update sensor data
            sensorData.ax1 = doc["ax1"] | 0.0;
            sensorData.ay1 = doc["ay1"] | 0.0;
            sensorData.az1 = doc["az1"] | 0.0;

            sensorData.emg1 = doc["emg1"] | 0;
         /*
            Serial.println("Data received from client:");
            Serial.print("ax1: "); Serial.println(sensorData.ax/1);
            Serial.print("ay1: "); Serial.println(sensorData.ay1);
            Serial.print("az1: "); Serial.println(sensorData.az1);

     
            Serial.print("emg1: "); Serial.println(sensorData.emg1); */
        } else {
            Serial.print("JSON parse error: ");
            Serial.println(error.c_str());
        }
    }
);
  server.onNotFound(notFound);

  server.begin();  // it will start webserver
}

void loop(void) {
  getSensorDataJSON();
} 