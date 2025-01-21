from PIL import Image
import cv2
from paddleocr import PaddleOCR
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
def write_frame_time(idx_list,file_name):
    with open(file_name ,'w') as f:
        for i in range(0,len(idx_list)):
            f.write(str(idx_list[i]) + "\n")


ocr = PaddleOCR(use_angle_cls=False, lang='chinese_cht',use_gpu=0) # need to run only once to load model into memory
# 設定影片路徑
video_path = "/home/kevin/whisperX/apple.mp4"

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 每秒幀數 (選用)
print(f"影片總幀數: {total_frames}")
print(f"影片 FPS: {fps}")
idx_list=[]
file_name='apple_idx.txt'
# 设置像素变化的阈值
change_threshold = 0.6  # SSIM 值小于此值时认为发生了变化
prev_region = None 
for idx in range(0, total_frames):  # 每幀去做
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    # 获取视频帧的高度和宽度
    height, width, _ = frame.shape

    # 裁剪下方文字区域（调整范围适配你的视频）
    cropped = frame[int(height * 0.8):int(height * 0.95), int(width * 0.05):int(width * 0.75)]

    # 转为灰度图像以简化计算
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # 將 BGR 格式轉為 RGB 格式，並轉為 PIL 圖片
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = Image.fromarray(frame)

    # 如果有前一帧，计算相似性
    if prev_region is not None:
        # 使用 SSIM 计算相似性
        score = ssim(prev_region, gray_cropped)

        # 如果变化显著（相似性低于阈值），保存帧
        if score < change_threshold:
            print(f"Frame {idx}: Significant change detected (SSIM: {score:.2f})")
            # output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            # cv2.imwrite(output_path, frame)
            idx_list.append(idx)

    # 更新前一帧区域
    prev_region = gray_cropped
    prev_frame  = frame

cap.release()
write_frame_time(idx_list,file_name)