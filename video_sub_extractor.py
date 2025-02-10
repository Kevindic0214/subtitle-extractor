import os
import re
import cv2
import ffmpeg
import logging
import numpy as np
from datetime import timedelta
from difflib import SequenceMatcher
from paddleocr import PaddleOCR

# 設定全域的 Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 若需要將日誌輸出到檔案，可自行新增 FileHandler
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# 初始化 PaddleOCR (建議在程式進入點就初始化，以免不斷重複初始化)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="chinese_cht",
    det_db_box_thresh=0.5,
    rec_algorithm="SVTR_LCNet",
    use_gpu=True
)

def format_time(seconds: float) -> str:
    """
    轉換秒數為 VTT 時間格式 (hh:mm:ss.sss)
    """
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=seconds)
    formatted = f"{td.seconds // 3600:02}:{(td.seconds % 3600) // 60:02}:{td.seconds % 60:02}.{int(td.microseconds / 1000):03}"
    return formatted

def normalize_text(text: str) -> str:
    """
    移除標點符號和空格以進行更準確的相似度比對
    """
    return re.sub(r'[\s\W]', '', text)

def similar(a: str, b: str) -> float:
    """
    計算兩個字串的相似度
    """
    return SequenceMatcher(None, a, b).ratio()

def merge_subtitles(subtitles, similarity_threshold=0.5):
    """
    合併相似的字幕區塊，並去除少於 4 個字的結果
    """
    logger.debug("開始合併字幕區塊 (merge_subtitles)")
    merged_subs = []
    for sub in subtitles:
        norm_sub_text = normalize_text(sub[2])
        # 跳過少於 4 個字的字幕
        if len(norm_sub_text) < 4:
            logger.debug(f"跳過字幕(長度 < 4): {sub[2]}")
            continue

        if merged_subs:
            norm_last_text = normalize_text(merged_subs[-1][2])
            if similar(norm_last_text, norm_sub_text) > similarity_threshold:
                # 更新結束時間並保持原始格式
                old_start, _, old_text = merged_subs[-1]
                merged_subs[-1] = (old_start, sub[1], old_text)
                logger.debug(f"合併相似字幕: {old_text} | {sub[2]}")
                continue
        
        merged_subs.append(sub)
    logger.debug(f"完成合併，共有 {len(merged_subs)} 筆字幕")
    return merged_subs

def generate_vtt(subtitles, output_path: str):
    """
    產生 VTT 字幕檔，並將相似且過短的區塊合併
    """
    logger.info(f"開始產生 VTT 檔: {output_path}")
    subtitles = merge_subtitles(subtitles)  # 先合併相似的字幕
    
    with open(output_path, "w", encoding="utf-8") as vtt:
        vtt.write("WEBVTT\nKind: captions\nLanguage: zh-TW\n\n")
        for start_time, end_time, text in subtitles:
            vtt.write(f"{start_time} --> {end_time}\n{text}\n\n")
    logger.info(f"VTT 字幕已產生完成: {output_path}")

def ocr_image(image_path: str) -> str:
    """
    讀取圖片並使用 PaddleOCR 辨識字幕，使用指定的座標區域：
    (y1=884, y2=1002, x1=204, x2=1727)
    """
    logger.debug(f"辨識圖片: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"無法讀取圖片或圖片不存在: {image_path}")
        return ""

    # 指定裁切範圍 (y1:y2, x1:x2)
    # y1=884, y2=1002, x1=204, x2=1727
    cropped_img = img[884:1002, 204:1727]

    result = ocr.ocr(cropped_img, cls=True)
    
    text_lines = []
    for line in result:
        if line:
            # line 內的每個元素形如 [位置, (文字, 置信度)]
            text_lines.append(" ".join([word[1][0] for word in line]))
    text = "\n".join(text_lines).strip()
    if text:
        logger.debug(f"OCR 結果: {text}")
    return text if text else ""

def process_frames(frames_folder: str, fps: float, start_time: float, time_adjustment: float = 0.0):
    """
    OCR 辨識影格並整理字幕資訊，可透過 time_adjustment 修正時間誤差
    """
    logger.info(f"開始處理影格資料夾: {frames_folder}")
    if not os.path.isdir(frames_folder):
        logger.error(f"指定的影格資料夾不存在: {frames_folder}")
        return []

    frame_files = sorted(os.listdir(frames_folder))
    subtitles = []
    frame_interval = 1 / fps

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, frame_file)
        text = ocr_image(frame_path)
        if text:  # 若有辨識到內容
            # 計算起始與結束時間
            start_sec = start_time + idx * frame_interval + time_adjustment
            end_sec = start_sec + frame_interval
            subtitles.append((format_time(start_sec), format_time(end_sec), text))

    logger.info(f"完成處理 {len(frame_files)} 張影格，產生 {len(subtitles)} 筆字幕")
    return subtitles

def extract_frames(video_path: str, output_folder: str, fps: int = 2, skip_start: int = 0, skip_end: int = 0):
    """
    使用 FFmpeg 抽取字幕畫面，根據 skip_start 與 skip_end 調整，若不想跳過，皆設為 0
    """
    logger.info(f"開始從影片擷取影格: {video_path}")
    logger.info(f"FPS={fps}, skip_start={skip_start}, skip_end={skip_end}")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        start_time = skip_start
        end_time = max(0, duration - skip_end)

        output_pattern = os.path.join(output_folder, "frame_%05d.png")
        logger.debug(f"抽影格時間區間: {start_time} 秒 ~ {end_time} 秒, 輸出路徑模式: {output_pattern}")

        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .output(output_pattern, vf=f"fps={fps}", vsync="vfr", **{'qscale:v': 2})
            .run()
        )
        logger.info("完成抽取影格")
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg 抽取影格時發生錯誤: {e}")
        raise

def process_video(video_path: str, output_vtt: str, fps: int = 2):
    """
    主流程：
    1. 使用 FFmpeg 抽影格（不跳過前後影片）。
    2. 使用 PaddleOCR 進行文字辨識 (指定區域)。
    3. 產生 VTT 字幕檔。
    """
    logger.info(f"準備處理影片: {video_path}")
    logger.info(f"輸出字幕: {output_vtt}, FPS={fps}")

    temp_folder = "frames"
    # 不跳過前後，所以 skip_start=0, skip_end=0
    extract_frames(video_path, temp_folder, fps=fps, skip_start=0, skip_end=0)
    # 開始辨識影格，此時因為沒有跳過，所以起始時間可以視需求自行給定(此處從0秒開始)
    subtitles = process_frames(temp_folder, fps, start_time=0.0, time_adjustment=0.0)
    generate_vtt(subtitles, output_vtt)

    logger.info(f"字幕檔已儲存至 {output_vtt}")

# 使用範例 (請自行移除或修改)
if __name__ == "__main__":
    video_file = "apple.mp4"
    vtt_output = "apple.vtt"
    process_video(video_file, vtt_output, fps=2)
