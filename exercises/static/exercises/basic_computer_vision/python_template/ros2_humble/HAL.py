import rclpy
import sys
import threading
import time
import cv2
import numpy as np
import json

from hal_interfaces.general.camera import CameraNode
from hal_interfaces.general.console import start_console
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# Constants
IMG_WIDTH = 640
IMG_HEIGHT = 480
MAX_CORNERS = 100

class ObjectTracker:
    def __init__(self):
        # ROS2 init
        if not rclpy.ok():
            rclpy.init(args=sys.argv)
        
        # Initialize nodes
        self.camera_node = CameraNode("/camera/image/compressed")
        
        # Spin nodes so that subscription callbacks load topic data
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.camera_node)
        self.executor_thread = threading.Thread(target=self.__auto_spin, daemon=True)
        self.executor_thread.start()
        
        # Tracking parameters
        self.feature_params = {
            'maxCorners': MAX_CORNERS,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }
        
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # Tracking state
        self.old_gray = None
        self.p0 = None
        self.mask = None
        self.frame = None
        
        # Console for parameter updates
        self.console = start_console()
        self.console.add_callback('tracking_params', self.update_tracking_params)
        
        # Start tracking loop
        self.tracking_thread = threading.Thread(target=self.tracking_loop, daemon=True)
        self.tracking_thread.start()
    
    def __auto_spin(self):
        while rclpy.ok():
            self.executor.spin_once(timeout_sec=0)
            time.sleep(1/90.0)  # Match typical ROS2 spin frequency
    
    def update_tracking_params(self, params):
        """Update tracking parameters from console"""
        try:
            params_dict = json.loads(params)
            for key, value in params_dict.items():
                if key in self.feature_params:
                    self.feature_params[key] = value
                elif key in self.lk_params:
                    self.lk_params[key] = value
        except json.JSONDecodeError:
            print("Invalid parameter format")
    
    def get_image(self):
        """Get latest image from camera node"""
        image = self.camera_node.getImage()
        while image is None:
            image = self.camera_node.getImage()
        return image.data
    
    def tracking_loop(self):
        """Main tracking loop"""
        while True:
            try:
                # Get image
                np_arr = np.frombuffer(self.get_image(), np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Initialize tracking if needed
                if self.old_gray is None:
                    self.old_gray = frame_gray.copy()
                    self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                    self.mask = np.zeros_like(frame)
                
                # Calculate optical flow
                if self.p0 is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
                    
                    # Select good points
                    if p1 is not None:
                        good_new = p1[st==1]
                        good_old = self.p0[st==1]
                        
                        # Draw the tracks
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                        
                        # Update the previous frame and points
                        self.old_gray = frame_gray.copy()
                        self.p0 = good_new.reshape(-1, 1, 2)
                
                # Combine mask and frame
                img = cv2.add(frame, self.mask)
                
                # Convert back to compressed format
                result, buffer = cv2.imencode('.jpg', img)
                if result:
                    self.console.send('tracking_result', buffer.tobytes())
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in tracking loop: {e}")
                time.sleep(1)  # Wait before retrying

if __name__ == "__main__":
    tracker = ObjectTracker()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down")