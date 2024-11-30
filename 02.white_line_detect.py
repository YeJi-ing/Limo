#!/usr/bin/env python3
'''
요약      : 이미지에서 흰색 선을 추출하는 코드
흐름      : 구독 → Bird-eye View 변환 → 흰색 선 추출 → 게시
[Topic] Subscribe : /camera/rgb/image_raw/compressed (콜백 함수: img_CB)
[Topic] Publish   : /white/compressed
[Function] img_warp     : Bird-eye view 변환된 이미지 반환
[Function] detect_color : 추출된 흰색 선 이미지 반환
'''
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class White_line_Detect:
    def __init__(self):
        self.bridge = CvBridge()           # CvBridge 객체 초기화
        rospy.init_node("white_line_node") # ROS 노드 초기화
        self.pub = rospy.Publisher("/white/compressed", CompressedImage, queue_size=10)    # Publisher 설정
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB) # Subscriber 설정

    def detect_color(self, img): # 흰색 선 탐지하는 함수
        # Convert to HSV color space: 이미지를 HSV color space로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range of white color in HSV: 흰색을 나타내는 HSV 범위 설정
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([179, 100, 255])

        # Threshold the HSV image to get only white colors
        white_mask = cv2.inRange(hsv, white_lower, white_upper)  # HSV 이미지에서 흰색 영역을 이진 마스크로 생성
        cv2.imshow("img", img)
        cv2.imshow("white_mask", white_mask)
        white_color = cv2.bitwise_and(img, img, mask=white_mask) # 흰색 마스크를 적용하여 원본 이미지에서 흰색만 추출
        cv2.imshow("white_color", white_color)
        return white_color # 추출된 흰색 선 이미지를 반환

    def img_warp(self, img): # Bird-eye view 변환하는 함수
        self.img_x, self.img_y = img.shape[1], img.shape[0]
        # print(f'self.img_x:{self.img_x}, self.img_y:{self.img_y}')

        img_size = [640, 480]
        # ROI
        src_side_offset = [0, 240]
        src_center_offset = [200, 315]
        src = np.float32(
            [
                [0, 479],
                [src_center_offset[0], src_center_offset[1]],
                [640 - src_center_offset[0], src_center_offset[1]],
                [639, 479],
            ]
        )
        # 아래 2 개 점 기준으로 dst 영역을 설정합니다.
        dst_offset = [round(self.img_x * 0.125), 0]
        # offset x 값이 작아질 수록 dst box width 증가합니다.
        dst = np.float32(
            [
                [dst_offset[0], self.img_y],
                [dst_offset[0], 0],
                [self.img_x - dst_offset[0], 0],
                [self.img_x - dst_offset[0], self.img_y],
            ]
        )
        # find perspective matrix
        matrix = cv2.getPerspectiveTransform(src, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(img, matrix, [self.img_x, self.img_y])
        return warp_img # 변환된 이미지 반환

    def img_CB(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data) # 수신된 CompressedImage 메시지를 OpenCV 이미지로 디코딩
        warp_img = self.img_warp(img)                    # Bird-eye View 변환 수행
        white_color = self.detect_color(warp_img)        # 변환된 이미지에서 흰색 선 탐지
        white_line_img_msg = self.bridge.cv2_to_compressed_imgmsg(white_color)  # 흰색 선 이미지를 ROS 메시지로 인코딩
        self.pub.publish(white_line_img_msg)             # 변환된 흰색 선 이미지를 ROS 토픽으로 게시
        
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("white_color", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        cv2.imshow("white_color", white_color)
        cv2.waitKey(1)


if __name__ == "__main__":
    white_line_detect = White_line_Detect()
    try:
        rospy.spin()                    # ROS 콜백 대기 (노드 종료될 때까지 실행 유지)
    except rospy.ROSInterruptException: # ROS 종료 예외 처리
        pass