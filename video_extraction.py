import cv2
import pytesseract
import logging
from tqdm import tqdm

# ====== logging 設定 ======
# 設定 logging 的等級、格式等
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ====== Tesseract 路徑設定 (依照實際安裝位置調整) ======
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
x_min, y_min = 0, 870
x_max, y_max = 1920, 990

# ====== 使用 tqdm 顯示進度 ======
for frame_num in tqdm(range(total_frames), desc='Processing frames'):
    ret, frame = cap.read()
    if not ret:
        logging.warning("影片讀取失敗或已結束。")
        break
    
    # 裁切出字幕區域
    subtitle_region = frame[y_min:y_max, x_min:x_max]
    
    # 進行灰階與二值化（依需求調整參數）
    gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # 執行 OCR
    subtitle = pytesseract.image_to_string(thresh, lang='chi_tra', config='--psm 6').strip()
    
    # 輸出幀數與字幕
    logging.info(f"第 {frame_num} 幀的辨識結果：'{subtitle}'")

cap.release()