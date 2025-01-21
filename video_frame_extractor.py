from PIL import Image
import cv2

# 設定影片路徑
video_path = 'example.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 每秒幀數 (選用)
print(f"影片總幀數: {total_frames}")
print(f"影片 FPS: {fps}")

# 初始化設定
skip_frame = 1  # 每幾幀處理一次

for idx in range(0, total_frames, skip_frame):  # 每幀去做
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    if not ret:
        print(f"無法讀取第 {idx} 幀")
        continue

    # 將 BGR 格式轉為 RGB 格式，並轉為 PIL 圖片
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    if idx == 238:  # 儲存第 5 幀
        output_path = f'/home/kevin/TinyLLaVA_Factory/frame_{idx}.jpg'
        frame.save(output_path)
        print(f"已儲存第 {idx} 幀到 {output_path}")
        break

cap.release()

        