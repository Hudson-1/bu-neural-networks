#include <ros.h>           //ROS header. (necessary for ROS programming using C++)
#include <geometry_msgs/Quaternion.h>//header including definition of type <geometry_msgs::Quaternion>.
#include <Adafruit_BNO08x.h>

   /*** Global Variables ***/
#define BNO08X_RESET -1
Adafruit_BNO08x  bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

ros::NodeHandle nh;        //node handle.

geometry_msgs::Quaternion qmsg;      //variable imsg with type <std_msgs::Int64>.

ros::Publisher pub("imu_topic",&qmsg);//a publisher.

/*** The setup() function is used to initialize the arduino board
like pin state and variable assignment ***/

void setup()  
{  
  
  nh.getHardware()->setBaud(9600);

  // Initialize the node
  nh.initNode();

  nh.advertise(pub);       //tell ROS to register a topic.

  if (!bno08x.begin_I2C()) {   
    while (1) { delay(10); }
  }
  qmsg.x=0;             //initialize qmsg.
  qmsg.y=0;
  qmsg.z=0;
  qmsg.w=0;
}


void loop(){
    bno08x.getSensorEvent(&sensorValue);
    switch (sensorValue.sensorId) {
      case SH2_GAME_ROTATION_VECTOR:
        qmsg.x = sensorValue.un.gameRotationVector.i;
        qmsg.y = sensorValue.un.gameRotationVector.j;
        qmsg.z = sensorValue.un.gameRotationVector.k;
        qmsg.w = sensorValue.un.gameRotationVector.real;
    }
  pub.publish(&qmsg);
  nh.spinOnce();
  delay(100);  
}
