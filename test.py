import os
import cv2
import logging
from tqdm import tqdm
from paddleocr import PaddleOCR

# ====== logging 設定 ======
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ====== 設定信心分數閾值 ======
confidence_threshold = 0.7  # 只有信心分數 >= 0.7 的結果會被納入

# ====== 初始化 PaddleOCR ======
ocr = PaddleOCR(lang='chinese_cht')  # 默認使用繁體中文模型

# ====== 開啟影片 ======
cap = cv2.VideoCapture('example.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

logging.info(f"影片幀率 (FPS): {fps}")
logging.info(f"總幀數 (Total Frames): {total_frames}")

# ====== 初始化變數 ======
previous_subtitle = None
subtitle_timings = []
start_frame = 0

# ====== 定義字幕區域座標 (可依實際情況調整) ======
x_min, y_min = 0, 890
x_max, y_max = 1920, 990

# ====== 建立輸出資料夾 (若不存在則自動建立) ======
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# 開啟一個文字檔來寫入資料
with open("ocr_output.txt", "w", encoding="utf-8") as f:
    # ====== 使用 tqdm 顯示進度 ======
    for frame_num in tqdm(range(total_frames), desc='Processing frames'):
        ret, frame = cap.read()
        if not ret:
            logging.warning("影片讀取失敗或已結束。")
            break

        # 裁切出字幕區域
        subtitle_region = frame[y_min:y_max, x_min:x_max]

        # 進行灰階與二值化（可視需要自行調整或移除）
        gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # 執行 OCR
        result = ocr.ocr(thresh, rec=True, cls=False)

        # 檢查 result 結構並提取文字
        if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            filtered_lines = []
            for line in result[0]:
                if isinstance(line, list) and len(line) > 1 and isinstance(line[1], tuple):
                    text, score = line[1]
                    if score >= confidence_threshold:
                        filtered_lines.append(text)
            subtitle = ' '.join(filtered_lines).strip()
        else:
            subtitle = ''

        # 在終端顯示過濾後的結果（只有高信心分數的文字）
        print(f"Frame {frame_num} OCR Subtitle: '{subtitle}'")

        # 將 frame id 以及字幕寫入文字檔，每行一筆
        f.write(f"{frame_num}\t{subtitle}\n")

        # 每 50 幀時保存處理後的圖像到指定資料夾
        if frame_num % 50 == 0:
            output_path = os.path.join(output_folder, f"subtitle_frame_{frame_num}.png")
            cv2.imwrite(output_path, thresh)

cap.release()
