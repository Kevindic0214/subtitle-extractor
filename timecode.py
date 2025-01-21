import re
from difflib import SequenceMatcher
from collections import Counter

# 分行解析
with open('/home/kevin/TinyLLaVA_Factory/apple.mp4_1.txt', 'r') as f:
    parsed_data = []
    for line in f:
        frame, text = line.strip().split(" ", 1)
        if(len(text) < 3):
            continue
        frame = int(frame)  # 將 frame 轉為整數
        # 過濾掉重複的 frame 和包含過多重複內容的文字
        parsed_data.append((frame, text))

# 相似度計算
def is_similar(text1, text2, threshold=0.25):
    """判斷兩段文字是否相似"""
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity >= threshold

# 分段處理 + 合併
segments = []
current_texts = [parsed_data[0][1]]
current_start = parsed_data[0][0]

for i in range(1, len(parsed_data)):
    frame, text = parsed_data[i]
    
    # 如果當前文字與前一段的文字相似，合併
    if any(is_similar(text, prev_text) for prev_text in current_texts):
        current_texts.append(text)
    else:
        # 不相似，結束當前段落，開始新段落
        most_common_text = Counter(current_texts).most_common(1)[0][0]
        segments.append((current_start, parsed_data[i - 1][0], most_common_text))
        current_start = frame
        current_texts = [text]

# 處理最後一段
most_common_text = Counter(current_texts).most_common(1)[0][0]
segments.append((current_start, parsed_data[-1][0], most_common_text))

# 輸出結果
with open('timecode.txt', 'w') as f:
    for start, end, text in segments:
        f.write(f"{start} 到 {end}: {text}" + "\n")

def frames_to_timestamp(frame, fps=30):
    """將 frame 數轉換為 SRT 格式的時間戳 (hh:mm:ss;ff)"""
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int((total_seconds - int(total_seconds)) * fps)
    return f"{hours:02}:{minutes:02}:{seconds:02};{frames:02}"

def process_ocr_to_subtitle(input_file, output_file, fps=30):
    """處理 OCR 框架並生成預期格式的字幕"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    subtitles = []
    current_text = None
    start_frame = None

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        frame, text = int(parts[0]), ' '.join(parts[1:])

        if text != current_text:
            if current_text is not None:
                # 保存前一段
                end_frame = frame - 1
                start_time = frames_to_timestamp(start_frame, fps)
                end_time = frames_to_timestamp(end_frame, fps)
                subtitles.append(f"{start_time} {end_time} {current_text}")
            # 開始新段落
            current_text = text
            start_frame = frame

    # 處理最後一段
    if current_text is not None:
        end_frame = frame
        start_time = frames_to_timestamp(start_frame, fps)
        end_time = frames_to_timestamp(end_frame, fps)
        subtitles.append(f"{start_time} {end_time} {current_text}")

    # 寫入輸出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(subtitles))
    print(f"字幕已保存至 {output_file}")

# 使用方法
input_txt = "apple.mp4_1.txt"  # 輸入的 txt 文件路徑
output_srt = "/home/kevin/TinyLLaVA_Factory/timecode.txt"  # 輸出的字幕文件路徑
process_ocr_to_subtitle(input_txt, output_srt)



import difflib

def parse_data(lines):
    # 提取 frame 和 OCR 文本
    parsed_data = []
    for line in lines:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            frame = int(parts[0])
            text = parts[1]
            parsed_data.append((frame, text))
    return parsed_data

def frame_to_timecode(frame, fps=30):
    # 轉換 frame 為時間碼 (hh:mm:ss;ff)
    seconds = frame / fps
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    ff = int((seconds - int(seconds)) * fps)
    return f"{hh:02}:{mm:02}:{ss:02};{ff:02}"

def merge_similar_texts(data, similarity_threshold=0.1, fps=30):
    merged_results = []
    start_frame = data[0][0]
    current_text = data[0][1]

    for i in range(1, len(data)):
        frame, text = data[i]
        similarity = difflib.SequenceMatcher(None, current_text, text).ratio()

        if similarity > similarity_threshold:
            # 如果相似，則繼續合併
            continue
        else:
            # 如果不相似，保存當前段
            end_frame = data[i-1][0]
            start_time = frame_to_timecode(start_frame, fps)
            end_time = frame_to_timecode(end_frame, fps)
            merged_results.append((start_time, end_time, current_text))
            # 更新新的段
            start_frame = frame
            current_text = text

    # 添加最後一段
    end_frame = data[-1][0]
    start_time = frame_to_timecode(start_frame, fps)
    end_time = frame_to_timecode(end_frame, fps)
    merged_results.append((start_time, end_time, current_text))

    return merged_results

def format_results(results):
    # 格式化結果為字符串
    return "\n".join([f"{start} {end} {text}" for start, end, text in results])

# 讀取輸入數據
with open("apple.mp4_1.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 處理流程
data = parse_data(lines)
merged_results = merge_similar_texts(data, similarity_threshold=0.7, fps=30)
output = format_results(merged_results)

# 輸出結果
with open("/home/kevin/TinyLLaVA_Factory/timecode.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("合併完成，結果已保存至 /home/kevin/TinyLLaVA_Factory/timecode.txt")