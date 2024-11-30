#!/usr/bin/env python3
'''
요약      : 슬라이딩 윈도우를 통해 차선(왼쪽, 오른쪽 차선의 2차 곡선) 인식하는 코드
흐름      : 구독 → Bird-eye View 변환 → 차선 인식 → 차선 위치 계산 → 게시
[Topic] Subscribe : /camera/rgb/image_raw/compressed (콜백 함수: img_CB)
[Topic] Publish   : /sliding_windows/compressed
[Class] Sliding_Window
    - [Function] detect_color   : 이미지에서 노란색과 흰색 차선 색상을 추출
    - [Function] img_warp       : Bird-eye view 변환된 이미지 반환 (왜곡 보정)
    - [Function] img_binary     : 색상 추출된 이미지를 이진화 (흑백 처리)
    - [Function] detect_nothing : 초기값 설정
    - [Function] window_search  : 차선 탐지
    - [Function] img_CB
'''
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


class Sliding_Window:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("sliding_window_node")
        self.pub = rospy.Publisher("/sliding_windows/compressed", CompressedImage, queue_size=10)
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB)
        self.nothing_flag = False

    def detect_color(self, img):
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range of yellow color in HSV: 노란색
        yellow_lower = np.array([0, 80, 0])
        yellow_upper = np.array([45, 255, 255])

        # Define range of blend color in HSV: 흰색
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([179, 64, 255])

        # Threshold the HSV image to get only yellow colors
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Threshold the HSV image to get only white colors
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # Threshold the HSV image to get blend colors
        blend_mask = cv2.bitwise_or(yellow_mask, white_mask)
        blend_color = cv2.bitwise_and(img, img, mask=blend_mask) # 원본 이미지에서 마스크를 이용해 색상만 추출
        return blend_color

    def img_warp(self, img, blend_color): 
        # 이미지를 왜곡 보정하여 Bird-eye view로 변환하는 함수
        self.img_x, self.img_y = img.shape[1], img.shape[0] # shape of img
        # print(f'self.img_x:{self.img_x}, self.img_y:{self.img_y}')
        # img_size = [640, 480]

        # ROI
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
        # 아래 2 개 점 기준으로 dst 영역을 설정
        dst_offset = [round(self.img_x * 0.125), 0]
        # offset x 값이 작아질 수록 dst box width 증가
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
        warp_img = cv2.warpPerspective(blend_color, matrix, [self.img_x, self.img_y])
        return warp_img

    def img_binary(self, blend_line): 
        # 색상 추출된 이미지를 그레이스케일로 변환 후 이진화 처리하는 함수
        bin = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY) # 이미지를 그레이스케일로 변환
        binary_line = np.zeros_like(bin)                   # 동일 크기의 이진화 이미지로 생성
        binary_line[bin != 0] = 1                          # 색상 있는 부분을 1로 설정
        return binary_line                                 # 이진화 이미지 반환

    def detect_nothing(self):
        # 첫 번째 이미지에서 차선이 검출되지 않았을 때 차선 위치를 초기화하는 함수
        self.nothing_left_x_base = round(self.img_x * 0.140625)
        self.nothing_right_x_base = self.img_x - round(self.img_x * 0.140625)

        self.nothing_pixel_left_x = np.zeros(self.nwindows) + round(self.img_x * 0.140625)
        self.nothing_pixel_right_x = np.zeros(self.nwindows) + self.img_x - round(self.img_x * 0.140625)
        self.nothing_pixel_y = [round(self.window_height / 2) * index for index in range(0, self.nwindows)]

    def window_search(self, binary_line):
        # 슬라이딩 윈도우 방식으로 차선의 위치를 추적하는 함수

        # histogram을 생성: y축 기준 절반 아래 부분만을 사용하여 x축 기준 픽셀의 분포 계산
        bottom_half_y = binary_line.shape[0] / 2
        histogram = np.sum(binary_line[int(bottom_half_y) :, :], axis=0) # 이미지 하단 부분의 히스토그램 계산
        # 히스토그램을 절반으로 나누어 좌우 히스토그램의 최대값의 인덱스를 반환
        midpoint = np.int32(histogram.shape[0] / 2)                      # 이미지의 중간값
        left_x_base = np.argmax(histogram[:midpoint])                    # 왼쪽 차선의 시작 위치
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint        # 오른쪽 차선의 시작 위치
        # show histogram
        # plt.hist(histogram)
        # plt.show()

        # 윈도우 탐색 시작
        if left_x_base == 0:
            left_x_current = self.nothing_left_x_base   # 차선이 없을 경우 기본값 설정
        else:
            left_x_current = left_x_base
        if right_x_base == midpoint:
            right_x_current = self.nothing_right_x_base # 차선이 없을 경우 기본값 설정
        else:
            right_x_current = right_x_base

        out_img = np.dstack((binary_line, binary_line, binary_line)) * 255 # 결과 저장할 이미지 생성

        ## window parameter
        nwindows = self.nwindows # 윈도우 개수 설정 (개수가 너무 적으면 정확X, 너무 많으면 연산량이 증가)
        window_height = self.window_height
        margin = 80              # 윈도우의 너비를 지정 (윈도우가 옆 차선까지 넘어가지 않게 사이즈를 적절히 지정)
        min_pix = round((margin * 2 * window_height) * 0.0031) # 탐색할 최소 픽셀의 개수를 지정

        # 차선의 픽셀 정보 추출
        lane_pixel = binary_line.nonzero()
        lane_pixel_y = np.array(lane_pixel[0])
        lane_pixel_x = np.array(lane_pixel[1])

        left_lane_idx = []   # 왼쪽 차선 픽셀 인덱스 담을 list
        right_lane_idx = []  # 오른쪽 차선 픽셀 인덱스 담을 list

        # 슬라이딩 윈도우로 차선 위치 추적
        for window in range(nwindows):
            # window boundary를 지정 (세로)
            win_y_low = binary_line.shape[0] - (window + 1) * window_height
            win_y_high = binary_line.shape[0] - window * window_height
            # print("check param : \n",window,win_y_low,win_y_high)

            # position 기준 window size
            win_x_left_low = left_x_current - margin
            win_x_left_high = left_x_current + margin
            win_x_right_low = right_x_current - margin
            win_x_right_high = right_x_current + margin

            # window 시각화
            if left_x_current != 0:
                cv2.rectangle(
                    out_img,
                    (win_x_left_low, win_y_low),
                    (win_x_left_high, win_y_high),
                    (0, 255, 0),
                    2,
                )
            if right_x_current != midpoint:
                cv2.rectangle(
                    out_img,
                    (win_x_right_low, win_y_low),
                    (win_x_right_high, win_y_high),
                    (0, 0, 255),
                    2,
                )

            # 왼쪽 오른쪽 각 차선 픽셀이 window안에 있는 경우 index를 저장
            good_left_idx = (
                (lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & (lane_pixel_x >= win_x_left_low) & (lane_pixel_x < win_x_left_high)
            ).nonzero()[0]
            good_right_idx = (
                (lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & (lane_pixel_x >= win_x_right_low) & (lane_pixel_x < win_x_right_high)
            ).nonzero()[0]

            # Append these indices to the lists
            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            # window내 설정한 pixel개수 이상이 탐지되면, 픽셀들의 x 좌표 평균으로 업데이트
            if len(good_left_idx) > min_pix:
                left_x_current = np.int32(np.mean(lane_pixel_x[good_left_idx]))
            if len(good_right_idx) > min_pix:
                right_x_current = np.int32(np.mean(lane_pixel_x[good_right_idx]))

        # np.concatenate(array) => axis 0으로 차원 감소 (window개수로 감소)
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        # window 별 좌우 도로 픽셀 좌표
        left_x = lane_pixel_x[left_lane_idx]
        left_y = lane_pixel_y[left_lane_idx]
        right_x = lane_pixel_x[right_lane_idx]
        right_y = lane_pixel_y[right_lane_idx]

        # 좌우 차선 별 2차 함수 계수를 추정
        if len(left_x) == 0 and len(right_x) == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
        else:
            if len(left_x) == 0:
                left_x = right_x - round(self.img_x / 2)
                left_y = right_y
            elif len(right_x) == 0:
                right_x = left_x + round(self.img_x / 2)
                right_y = left_y

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        # 좌우 차선 별 추정할 y좌표
        plot_y = np.linspace(0, binary_line.shape[0] - 1, 100)
        # 좌우 차선 별 2차 곡선을 추정
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        center_fit_x = (right_fit_x + left_fit_x) / 2

        # # window안의 lane을 black 처리
        # out_img[lane_pixel_y[left_lane_idx], lane_pixel_x[left_lane_idx]] = (0, 0, 0)
        # out_img[lane_pixel_y[right_lane_idx], lane_pixel_x[right_lane_idx]] = (0, 0, 0)

        # 양쪽 차선 및 중심 선 pixel 좌표(x,y)로 변환
        center = np.asarray(tuple(zip(center_fit_x, plot_y)), np.int32)
        right = np.asarray(tuple(zip(right_fit_x, plot_y)), np.int32)
        left = np.asarray(tuple(zip(left_fit_x, plot_y)), np.int32)

        cv2.polylines(out_img, [left], False, (0, 0, 255), thickness=5)
        cv2.polylines(out_img, [right], False, (0, 255, 0), thickness=5)
        sliding_window_img = out_img
        return sliding_window_img, left, right, center, left_x, left_y, right_x, right_y

    def img_CB(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        self.nwindows = 10
        self.window_height = np.int32(img.shape[0] / self.nwindows)
        blend_img = self.detect_color(img)
        warp_img = self.img_warp(img, blend_img)
        binary_img = self.img_binary(warp_img)
        if self.nothing_flag == False:
            self.detect_nothing()
            self.nothing_flag = True
        (
            sliding_window_img,
            left,
            right,
            center,
            left_x,
            left_y,
            right_x,
            right_y,
        ) = self.window_search(binary_img)

        os.system("clear")
        print(f"------------------------------")
        print(f"left : {left}")
        print(f"right : {right}")
        print(f"center : {center}")
        print(f"left_x : {left_x}")
        print(f"left_y : {left_y}")
        print(f"right_x : {right_x}")
        print(f"right_y : {right_y}")
        print(f"------------------------------")
        sliding_window_msg = self.bridge.cv2_to_compressed_imgmsg(sliding_window_img)
        self.pub.publish(sliding_window_msg)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("sliding_window_img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.imshow("sliding_window_img", sliding_window_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    sliding_window = Sliding_Window()
    rospy.spin()