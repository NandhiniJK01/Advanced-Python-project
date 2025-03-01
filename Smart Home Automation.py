import RPi.GPIO as GPIO
import time
import paho.mqtt.client as mqtt
import json

# GPIO Setup for Raspberry Pi
LIGHT_PIN = 17
FAN_PIN = 27
DOOR_PIN = 22
TEMP_SENSOR_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(LIGHT_PIN, GPIO.OUT)
GPIO.setup(FAN_PIN, GPIO.OUT)
GPIO.setup(DOOR_PIN, GPIO.OUT)

# MQTT Setup
BROKER = "mqtt.eclipse.org"
TOPIC_LIGHT = "home/light"
TOPIC_FAN = "home/fan"
TOPIC_DOOR = "home/door"
TOPIC_TEMP = "home/temperature"

def read_temperature():
    # Mock temperature reading (replace with actual sensor code)
    return 25.0

def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8").lower()
    try:
        data = json.loads(payload)
        if message.topic == TOPIC_LIGHT:
            GPIO.output(LIGHT_PIN, GPIO.HIGH if data['state'] == "on" else GPIO.LOW)
            print(f"Light turned {data['state']}")
        elif message.topic == TOPIC_FAN:
            GPIO.output(FAN_PIN, GPIO.HIGH if data['state'] == "on" else GPIO.LOW)
            print(f"Fan turned {data['state']}")
        elif message.topic == TOPIC_DOOR:
            GPIO.output(DOOR_PIN, GPIO.HIGH if data['state'] == "open" else GPIO.LOW)
            print(f"Door {data['state']}")
    except json.JSONDecodeError:
        print("Invalid JSON format received")

client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER)
client.subscribe([(TOPIC_LIGHT, 0), (TOPIC_FAN, 0), (TOPIC_DOOR, 0), (TOPIC_TEMP, 0)])
client.loop_start()

try:
    while True:
        temp = read_temperature()
        client.publish(TOPIC_TEMP, json.dumps({"temperature": temp}))
        print(f"Temperature: {temp}°C")
        time.sleep(5)
except KeyboardInterrupt:
    GPIO.cleanup()
    client.loop_stop()
    print("Smart Home Automation Stopped")
