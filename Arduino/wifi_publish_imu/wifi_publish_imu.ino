// Include the WiFi library
#include <WiFi.h>
#include <WiFiUdp.h>

// Define the WiFi network credentials
char ssid[] = "Hudson's iPhone 7"; // your network SSID (name)
char pass[] = "Aibel340"; // your network password

// Define the IP address and port of the ROS 2 serial bridge
IPAddress server(127,0,0,1); // change this to your ROS 2 serial bridge's IP
int port = 11411; // change this to your ROS 2 serial bridge's port

// Define a WiFi UDP object
WiFiUDP udp;

// Define a WiFiHardware class that implements the methods for rosserial communication
class WiFiHardware {
  public:
    WiFiHardware() {}; // constructor
    void init() { // initialize the WiFi connection and the UDP socket
      // Connect to the WiFi network
      WiFi.begin(ssid, pass);
      while (WiFi.status() != WL_CONNECTED) {
        delay(500);
      }
      // Begin the UDP socket
      udp.begin(port);
    }
    int read() { // read a byte from the UDP socket
      return udp.read(); // return -1 if there is an error or no data available
    }
    void write(uint8_t* data, int length) { // write data to the UDP socket
      udp.beginPacket(server, port); // start a packet to send to the ROS 2 serial bridge
      udp.write(data, length); // write the data
      udp.endPacket(); // end and send the packet
    }
    unsigned long time() { // return the current time in milliseconds
      return millis();
    }
    void setBaud(long baud) { // set the baud rate for the serial communication
  // implement this method according to your hardware specifications
  // for example, if you use a Serial1 object, you can use this code:
  Serial1.begin(baud);
}

};

// Include the ros.h and geometry_msgs/Quaternion.h headers as before
#include <ros.h>
#include <geometry_msgs/Quaternion.h>

// Include the Adafruit_BNO08x header as before
#include <Adafruit_BNO08x.h>

// Define the BNO08X_RESET pin as before
#define BNO08X_RESET -1

// Create a BNO08x object as before
Adafruit_BNO08x  bno08x(BNO08X_RESET);

// Create a sensorValue object as before
sh2_SensorValue_t sensorValue;

// Create a node handle object with the WiFiHardware class as template parameter
ros::NodeHandle_<WiFiHardware> nh;

// Create a quaternion message object as before
geometry_msgs::Quaternion qmsg;

// Create a publisher object as before
ros::Publisher pub("imu_topic",&qmsg);

void setup() {
  nh.getHardware()->setBaud(9600); // set the baud rate as before

  nh.initNode(); // initialize the node as before

  nh.advertise(pub); // advertise the publisher as before

  if (!bno08x.begin_I2C()) { // initialize the BNO08x sensor as before   
    while (1) { delay(10); }
  }

  qmsg.x=0; // initialize the quaternion message as before            
  qmsg.y=0;
  qmsg.z=0;
  qmsg.w=0;
}

void loop() {
  bno08x.getSensorEvent(&sensorValue); // get the sensor event as before
  switch (sensorValue.sensorId) {
    case SH2_GAME_ROTATION_VECTOR: // update the quaternion message with the game rotation vector values as before
      qmsg.x = sensorValue.un.gameRotationVector.i;
      qmsg.y = sensorValue.un.gameRotationVector.j;
      qmsg.z = sensorValue.un.gameRotationVector.k;
      qmsg.w = sensorValue.un.gameRotationVector.real;
  }
  pub.publish(&qmsg); // publish the message as before
  nh.spinOnce(); // spin the node once as before
  delay(100); // delay for 100 ms as before 
}
