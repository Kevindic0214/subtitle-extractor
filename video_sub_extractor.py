import os
import re
import cv2
import ffmpeg
import logging
import time
from datetime import timedelta
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor, as_completed

# 設定全域 Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# 初始化 PaddleOCR (建議在程式進入點就初始化，以免重複初始化)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="chinese_cht",
    det_db_box_thresh=0.5,
    rec_algorithm="SVTR_LCNet",
    use_gpu=True
)

def format_time(seconds: float) -> str:
    """
    將秒數轉換為 VTT 時間格式 (hh:mm:ss.sss)
    """
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=seconds)
    # 注意：td.seconds 為一天內的秒數，若影片超過 24 小時則需要另外處理
    formatted = f"{td.seconds // 3600:02}:{(td.seconds % 3600) // 60:02}:{td.seconds % 60:02}.{int(td.microseconds / 1000):03}"
    return formatted

def normalize_text(text: str) -> str:
    """
    移除標點符號和空格以進行更準確的比對
    """
    return re.sub(r'[\s\W]', '', text)

def similar(a: str, b: str) -> float:
    """
    計算兩字串相似度
    """
    return SequenceMatcher(None, a, b).ratio()

def merge_subtitles(subtitles, similarity_threshold=0.5):
    """
    合併相似的字幕區塊，並跳過少於 4 個字的內容
    """
    logger.debug("開始合併字幕區塊 (merge_subtitles)")
    merged_subs = []
    for sub in subtitles:
        norm_sub_text = normalize_text(sub[2])
        if len(norm_sub_text) < 4:
            logger.debug(f"跳過字幕(長度 < 4): {sub[2]}")
            continue

        if merged_subs:
            norm_last_text = normalize_text(merged_subs[-1][2])
            if similar(norm_last_text, norm_sub_text) > similarity_threshold:
                # 更新上一筆字幕的結束時間，保留原始字幕內容
                old_start, _, old_text = merged_subs[-1]
                merged_subs[-1] = (old_start, sub[1], old_text)
                logger.debug(f"合併相似字幕: {old_text} | {sub[2]}")
                continue

        merged_subs.append(sub)
    logger.debug(f"完成合併，共有 {len(merged_subs)} 筆字幕")
    return merged_subs

def generate_vtt(subtitles, output_path: str):
    """
    產生 VTT 字幕檔，並將相似或過短的字幕進行合併
    """
    logger.info(f"開始產生 VTT 檔: {output_path}")
    subtitles = merge_subtitles(subtitles)
    try:
        with open(output_path, "w", encoding="utf-8") as vtt:
            vtt.write("WEBVTT\nKind: captions\nLanguage: zh-TW\n\n")
            for start_time, end_time, text in subtitles:
                vtt.write(f"{start_time} --> {end_time}\n{text}\n\n")
        logger.info(f"VTT 字幕產生完成: {output_path}")
    except Exception as e:
        logger.error(f"產生 VTT 檔時發生錯誤: {e}")

def ocr_image(image_path: str, crop_area: tuple) -> str:
    """
    讀取圖片並使用 PaddleOCR 辨識字幕，支援傳入裁切區域 (格式： (y1, y2, x1, x2))
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"無法讀取圖片或圖片不存在: {image_path}")
            return ""
    except Exception as e:
        logger.error(f"讀取圖片時發生錯誤 {image_path}: {e}")
        return ""

    try:
        if crop_area is not None:
            y1, y2, x1, x2 = crop_area
            cropped_img = img[y1:y2, x1:x2]
        else:
            cropped_img = img

        result = ocr.ocr(cropped_img, cls=True)
        text_lines = []
        for line in result:
            if line:
                # 每個 line 內的元素形如 [位置, (文字, 置信度)]
                text_lines.append(" ".join([word[1][0] for word in line]))
        text = "\n".join(text_lines).strip()
        if text:
            logger.debug(f"OCR 結果: {text}")
        return text if text else ""
    except Exception as e:
        logger.error(f"OCR 辨識失敗 {image_path}: {e}")
        return ""

def process_single_frame(frame_path: str, idx: int, fps: float, start_time: float, time_adjustment: float, crop_area: tuple):
    """
    處理單一影格的 OCR 與時間計算，回傳 (idx, start_time_str, end_time_str, text)
    """
    try:
        text = ocr_image(frame_path, crop_area=crop_area)
        if text:
            frame_interval = 1 / fps
            # 使用影片原始起始時間（經 skip_start 調整後）+ 當前影格間隔
            start_sec = start_time + idx * frame_interval + time_adjustment
            end_sec = start_sec + frame_interval
            return (idx, format_time(start_sec), format_time(end_sec), text)
    except Exception as e:
        logger.error(f"處理影格 {frame_path} 時發生錯誤: {e}")
    return None

def process_frames(frames_folder: str, fps: float, start_time: float, time_adjustment: float,
                   crop_area: tuple, max_workers: int):
    """
    OCR 辨識影格並整理字幕資訊，使用多執行緒平行處理以提升效能
    """
    logger.info(f"開始處理影格資料夾: {frames_folder}")
    if not os.path.isdir(frames_folder):
        logger.error(f"指定的影格資料夾不存在: {frames_folder}")
        return []

    frame_files = sorted(os.listdir(frames_folder))
    subtitles = []
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_folder, frame_file)
            future = executor.submit(process_single_frame, frame_path, idx, fps, start_time, time_adjustment, crop_area)
            futures.append(future)
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                subtitles.append(res)
    # 根據原始索引排序
    subtitles.sort(key=lambda x: x[0])
    # 移除排序用的 index，只保留 (start_time, end_time, text)
    final_subtitles = [(sub[1], sub[2], sub[3]) for sub in subtitles]
    logger.info(f"完成處理 {len(frame_files)} 張影格，產生 {len(final_subtitles)} 筆字幕")
    return final_subtitles

def extract_frames(video_path: str, output_folder: str, fps: int, skip_start: int, skip_end: int):
    """
    使用 FFmpeg 抽取影格，根據 skip_start 與 skip_end 調整擷取區間
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
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info("完成抽取影格")
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg 抽取影格時發生錯誤: {e.stderr.decode() if e.stderr else e}")
        raise
    except Exception as e:
        logger.error(f"抽取影格過程中發生未知錯誤: {e}")
        raise

def process_video(video_path: str, output_vtt: str, fps: int, skip_start: int, skip_end: int,
                  crop_area: tuple, time_adjustment: float, max_workers: int):
    """
    主流程：
      1. 使用 FFmpeg 根據指定參數抽取影格。
      2. 從影片資訊取得原始起始時間與幀率（用於更精確時間轉換）。
      3. 使用 PaddleOCR 辨識影格文字。
      4. 產生 VTT 字幕檔。
    """
    logger.info(f"準備處理影片: {video_path}")
    logger.info(f"輸出字幕: {output_vtt}, 擷取 FPS={fps}")
    
    temp_folder = "frames"
    # 抽取影格前取得影片資訊
    try:
        probe = ffmpeg.probe(video_path)
        video_duration = float(probe['format']['duration'])
        # 取得影片起始時間，若無則預設 0
        video_start_str = probe['streams'][0].get('start_time', '0')
        video_start_time = float(video_start_str)
        # 取得原始影片幀率
        r_frame_rate = probe['streams'][0].get('r_frame_rate', '0/0')
        try:
            num, den = r_frame_rate.split('/')
            original_fps = float(num) / float(den) if float(den) != 0 else 0
        except Exception:
            original_fps = 0
        logger.info(f"原影片起始時間: {video_start_time} 秒, 原影片幀率: {original_fps}")
    except Exception as e:
        logger.error(f"取得影片資訊失敗: {e}")
        video_start_time = 0

    # 抽取影格 (擷取區間會自動以 skip_start 與 skip_end 調整)
    extract_frames(video_path, temp_folder, fps=fps, skip_start=skip_start, skip_end=skip_end)
    # 以影片起始時間（加上 skip_start）作為 OCR 計算的基準時間
    extraction_start_time = video_start_time + skip_start
    subtitles = process_frames(temp_folder, fps, start_time=extraction_start_time,
                               time_adjustment=time_adjustment, crop_area=crop_area, max_workers=max_workers)
    generate_vtt(subtitles, output_vtt)
    logger.info(f"字幕檔已儲存至 {output_vtt}")

if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    video_file = "apple.mp4"
    vtt_output = "apple.vtt"
    # 可依需求調整參數，例如 fps、skip_start、skip_end、裁切區域、time_adjustment 與平行處理數量
    process_video(video_file, vtt_output, fps=2, skip_start=0, skip_end=0,
                  crop_area=(884, 1002, 204, 1727), time_adjustment=0.0, max_workers=4)
    end_time = time.time()    # 記錄結束時間
    logger.info(f"整支程式執行總時間: {end_time - start_time:.2f} 秒")
