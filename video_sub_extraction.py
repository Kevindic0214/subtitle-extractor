import cv2
import pytesseract
import pandas as pd
import os
from tqdm import tqdm
import math

def extract_subtitles(video_path, tesseract_cmd=None, subtitle_region_ratio=0.2, frame_interval=0.5, language='chi_tra'):
    """
    使用 OCR 從影片檔案中提取字幕。

    :param video_path: 影片檔案路徑。
    :param tesseract_cmd: Tesseract 可執行檔路徑。如為 None，假設已加入系統 PATH。
    :param subtitle_region_ratio: 影片高度中作為字幕區域的比例。
    :param frame_interval: 處理幀的時間間隔（秒）。
    :param language: Tesseract OCR 的語言參數。
    :return: 包含 'start_time'、'end_time' 和 'text' 的 DataFrame。
    """
    # 設定 Tesseract 命令路徑
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # 檢查影片檔案是否存在
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"找不到影片檔案：{video_path}")

    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法打開影片檔案：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    results = []
    last_text = None
    subtitle_start = None

    current_time = 0.0
    frame_number = 0

    # 計算總處理步數
    total_steps = math.ceil(duration / frame_interval)

    # 使用 tqdm 顯示進度條
    with tqdm(total=total_steps, desc="處理字幕", unit="frame") as pbar:
        while current_time < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  # 設定位置為毫秒
            ret, frame = cap.read()
            if not ret:
                break

            # 定義字幕區域
            height, width, _ = frame.shape
            y_start = int(height * (1 - subtitle_region_ratio))
            subtitle_region = frame[y_start:height, :]

            # 圖像預處理以提升 OCR 準確度
            gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
            # 可選：應用額外的預處理，如膨脹、腐蝕等
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # OCR 配置
            config = '--psm 6'  # 假設為單一均勻的文字區塊

            # 執行 OCR
            try:
                text = pytesseract.image_to_string(thresh, lang=language, config=config).strip()
            except pytesseract.TesseractError as e:
                print(f"OCR 錯誤於 {current_time:.2f}s：{e}")
                text = ""

            if text:
                if text != last_text:
                    if last_text is not None:
                        # 上一個字幕的結束時間
                        results.append({
                            'start_time': subtitle_start,
                            'end_time': current_time,
                            'text': last_text
                        })
                    # 新字幕的開始時間
                    subtitle_start = current_time
                    last_text = text
            else:
                if last_text is not None:
                    # 上一個字幕的結束時間
                    results.append({
                        'start_time': subtitle_start,
                        'end_time': current_time,
                        'text': last_text
                    })
                    last_text = None

            # 時間增加
            current_time += frame_interval
            frame_number += 1

            # 更新進度條
            pbar.update(1)

    # 處理影片結束時仍有字幕的情況
    if last_text is not None:
        results.append({
            'start_time': subtitle_start,
            'end_time': current_time,
            'text': last_text
        })

    cap.release()

    df = pd.DataFrame(results)
    return df

def save_to_srt(df, srt_path):
    """
    將字幕 DataFrame 保存為 SRT 檔案。

    :param df: 包含 'start_time'、'end_time' 和 'text' 的 DataFrame。
    :param srt_path: SRT 檔案的保存路徑。
    """
    def format_timestamp(seconds):
        millis = int(round((seconds - int(seconds)) * 1000))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            f.write(f"{idx + 1}\n")
            start = format_timestamp(row['start_time'])
            end = format_timestamp(row['end_time'])
            f.write(f"{start} --> {end}\n")
            f.write(f"{row['text']}\n\n")

if __name__ == "__main__":
    video_path = "example.mp4"
    srt_output = "subtitles.srt"
    # 根據您的安裝情況更新 Tesseract 路徑，若已加入 PATH，則設為 None
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows 範例
    try:
        subtitles = extract_subtitles(
            video_path,
            tesseract_cmd=tesseract_path,
            subtitle_region_ratio=0.2,
            frame_interval=0.5,
            language='chi_tra'  # 繁體中文
        )
        print(subtitles)
        save_to_srt(subtitles, srt_output)
        print(f"字幕已保存至 {srt_output}")
    except Exception as e:
        print(f"發生錯誤：{e}")
