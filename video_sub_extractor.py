import cv2
import paddleocr
import ffmpeg
import os
import numpy as np
from datetime import timedelta
from difflib import SequenceMatcher
import re

ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="chinese_cht", det_db_box_thresh=0.5, rec_algorithm="SVTR_LCNet", use_gpu=True)

def format_time(seconds):
    """ 轉換秒數為 VTT 時間格式 (hh:mm:ss.sss) """
    td = timedelta(seconds=seconds)
    return f"{td.seconds//3600:02}:{(td.seconds%3600)//60:02}:{td.seconds%60:02}.{int(td.microseconds/1000):03}"

def normalize_text(text):
    """ 移除標點符號和空格以進行更準確的相似度比對 """
    return re.sub(r'[\s\W]', '', text)

def similar(a, b):
    """ 計算兩個字串的相似度 """
    return SequenceMatcher(None, a, b).ratio()

def merge_subtitles(subtitles, similarity_threshold=0.5):
    """ 合併相似的字幕區塊，並去除少於4個字的結果 """
    merged_subs = []
    
    for sub in subtitles:
        norm_sub_text = normalize_text(sub[2])
        if len(norm_sub_text) < 4:
            continue  # 跳過少於4個字的字幕
        
        if merged_subs:
            norm_last_text = normalize_text(merged_subs[-1][2])
            if similar(norm_last_text, norm_sub_text) > similarity_threshold:
                # 更新結束時間並保持原始格式
                merged_subs[-1] = (merged_subs[-1][0], sub[1], merged_subs[-1][2])
                continue
        merged_subs.append(sub)
    
    return merged_subs

def generate_vtt(subtitles, output_path):
    """ 產生 VTT 字幕檔 """
    subtitles = merge_subtitles(subtitles)  # 先合併相似的字幕
    
    with open(output_path, "w", encoding="utf-8") as vtt:
        vtt.write("WEBVTT\nKind: captions\nLanguage: zh-TW\n\n")
        for start_time, end_time, text in subtitles:
            vtt.write(f"{start_time} --> {end_time}\n{text}\n\n")

def ocr_image(image_path):
    """ 讀取圖片並使用 PaddleOCR 辨識字幕（僅處理下方區域） """
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    cropped_img = img[int(height * 0.75):int(height * 0.95), int(width * 0.10):int(width * 0.90)]  # 取下方 25%-5% 且寬度 10%-90%
    result = ocr.ocr(cropped_img, cls=True)
    text_lines = []
    for line in result:
        if line:
            text_lines.append(" ".join([word[1][0] for word in line]))
    text = "\n".join(text_lines).strip()
    return text if text else None

def process_frames(frames_folder, fps, start_time, time_adjustment=-0.3):
    """ OCR 辨識影格並整理字幕資訊，並修正時間誤差 """
    frame_files = sorted(os.listdir(frames_folder))
    subtitles = []
    frame_interval = 1 / fps
    
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, frame_file)
        text = ocr_image(frame_path)
        if text:
            start_sec = start_time + (idx) * frame_interval + time_adjustment
            end_sec = start_sec + frame_interval
            subtitles.append((format_time(start_sec), format_time(end_sec), text))
    
    return subtitles

def extract_frames(video_path, output_folder, fps=2, skip_start=180, skip_end=180):
    """ 使用 FFmpeg 抽取字幕畫面，排除前後 3 分鐘 """
    os.makedirs(output_folder, exist_ok=True)
    
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    start_time = skip_start
    end_time = max(0, duration - skip_end)
    
    output_pattern = os.path.join(output_folder, "frame_%05d.png")
    (
        ffmpeg
        .input(video_path, ss=start_time, to=end_time)
        .output(output_pattern, vf=f"fps={fps}", vsync="vfr")
        .run()
    )

def process_video(video_path, output_vtt, fps=2):
    """ 主流程：抽影格、OCR 辨識、產生 VTT """
    temp_folder = "frames"
    extract_frames(video_path, temp_folder, fps=fps)
    subtitles = process_frames(temp_folder, fps, start_time=180, time_adjustment=-0.3)
    generate_vtt(subtitles, output_vtt)
    print(f"字幕檔已儲存至 {output_vtt}")

# 使用範例
video_file = "apple.mp4"
vtt_output = "apple.vtt"
process_video(video_file, vtt_output)