#!/usr/bin/env python3
'''
요약      : 이미지를 OpenCV 형식으로 변환 후 Bird-eye view 변환 수행(원근 변환)하여 게시하는 코드
흐름      : 구독 → 이미지 디코딩 → Bird-eye view 변환 → 이미지 인코딩 → 게시
[Topic] Subscribe : /camera/rgb/image_raw/compressed (콜백 함수: img_CB)
[Topic] Publish   : /bird_eye/compressed
[Function] img_warp : Bird-eye view 변환된 이미지 반환
'''
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2


class Bird_Eye_View:
    def __init__(self):
        self.bridge = CvBridge()          # CvBridge(ROS 이미지 메시지와 OpenCV 이미지 간 변환 도구) 초기화
        rospy.init_node("bird_eye_node0") # ROS 노드 초기화
        self.pub = rospy.Publisher(       # Publisher 설정
            "/bird_eye/compressed", CompressedImage, queue_size=10
        )
        rospy.Subscriber(                 # Subscriber 설정
            "/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB
        )

    def img_warp(self, img): # Bird-eye view 변환하는 함수
        self.img_x, self.img_y = img.shape[1], img.shape[0] # 이미지 가로, 세로 크기 추출
        # print(f'self.img_x:{self.img_x}, self.img_y:{self.img_y}')

        img_size = [640, 480]                               # 출력 이미지 크기 설정
        # ROI
        src_side_offset = [0, 240]                          # 원본 이미지의 ROI 좌표 설정
        src_center_offset = [200, 315]                      # 원본 이미지의 ROI 좌표 설정
        src = np.float32(
            [
                [0, 479],                                           # 좌측 하단 
                [src_center_offset[0], src_center_offset[1]],       # 좌측 상단 
                [640 - src_center_offset[0], src_center_offset[1]], # 우측 상단
                [639, 479],                                         # 우측 하단
            ]
        )
        # 변환 후 매핑될 출력 이미지의 좌표 설정
        dst_offset = [round(self.img_x * 0.125), 0]                           # 2개 점 기준으로 dst 영역을 설정, offset x 값이 작아질 수록 dst box width 증가
        dst = np.float32(
            [
                [dst_offset[0], self.img_y],
                [dst_offset[0], 0],
                [self.img_x - dst_offset[0], 0],
                [self.img_x - dst_offset[0], self.img_y],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)                        # 원근 변환 매트릭스 계산
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(img, matrix, [self.img_x, self.img_y]) # 원근 변환 적용하여 Bird-eye View 이미지 생성
        return warp_img # 변환된 이미지 반환

    def img_CB(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data)              # 수신된 CompressedImage 메시지를 OpenCV 이미지로 디코딩
        warp_img = self.img_warp(img)                                 # Bird-eye View 변환 수행
        warp_img_msg = self.bridge.cv2_to_compressed_imgmsg(warp_img) # 변환된 이미지를 CompressedImage 메시지로 다시 인코딩
        self.pub.publish(warp_img_msg)                                # 변환된 이미지를 publish
        
        # 원본 이미지와 변환된 이미지를 OpenCV 창에 표시
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("img_warp", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)           # 원본 이미지 
        cv2.imshow("img_warp", warp_img) # 변환된 이미지
        cv2.waitKey(1)                   # OpenCV 창 갱신


if __name__ == "__main__":
    bird_eye_view = Bird_Eye_View()
    rospy.spin() # 노드가 계속 동작하도록 ROS 이벤트 루프 유지