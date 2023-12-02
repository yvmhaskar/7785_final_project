
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
import numpy

class RotateSubPub(Node):
	def __init__(self):
		super().__init__('rotate_sub_pub')
		self.subscription = self.create_subscription(Int32, 'direction', self.listener_callback, 5)
		self.subscription
		self.cmd_vel = self.create_publisher(Twist,'cmd_vel',10)

	def listener_callback(self, msg):
		avg_speed = (numpy.pi)/4
		turn_dir = float(msg.data)
		twist = Twist()
		twist.angular.z = avg_speed*turn_dir
		self.cmd_vel.publish(twist)

#class VelPublisher(Node):
#   def __init__(self):
#      super().__init__('vel_publisher')
#      self.publisher_ = self.create_publisher(Twist,'cmd_vel', 5)
#      #self.timer = self.create_timer(0.5, self.publish_command)
#      self.twist_msg = Twist()

#   def publish_command(self):
#      self.twist_msg.angular.x = 0
#      self.twist_msg.angular.y = 0
#      self.twist_msg.angular.z = 1 # this value needs to change to somehow account for input
#      self.publisher_.publish(self.twist_msg)


def main(args=None):
	rclpy.init(args=args)
	# Subscriber
	rotate_sub_pub = RotateSubPub()
	rclpy.spin(rotate_sub_pub)

	rotate_sub_pub.destroy_node()


	# publisher
#    vel_publisher=VelPublisher()
#    rclpy.spin(vel_publisher)
#    vel_publisher.destroy_node()

	rclpy.shutdown()

if __name__ == '__main__':
	main()
