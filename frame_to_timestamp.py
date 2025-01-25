import difflib

# 設定影片參數
TOTAL_FRAMES = 19731  # 總幀數
FPS = 29              # 每秒影格數

# 讀取 ocr_output.txt 並解析成幀數與字幕的列表
def read_ocr_output(file_path):
    frames = []
    subtitles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # 忽略格式錯誤的行
            frame_num, subtitle = parts
            frames.append(int(frame_num))
            subtitles.append(subtitle)
    return frames, subtitles

# 將幀數轉換為時間（格式：時:分:秒,毫秒）
def frame_to_timestamp(frame, fps):
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# 合併相似字幕並生成時間戳
def merge_subtitles(frames, subtitles, similarity_threshold=0.8):
    groups = []
    if not subtitles:
        return groups

    # 初始化第一個分組
    current_group = {
        'start_frame': frames[0],
        'end_frame': frames[0],
        'subtitle': subtitles[0]
    }

    for i in range(1, len(subtitles)):
        prev_sub = current_group['subtitle']
        current_sub = subtitles[i]
        
        # 計算相似度
        similarity = difflib.SequenceMatcher(None, prev_sub, current_sub).ratio()
        
        if similarity >= similarity_threshold:
            # 合併到當前分組
            current_group['end_frame'] = frames[i]
        else:
            # 結束當前分組並開始新的分組
            groups.append(current_group)
            current_group = {
                'start_frame': frames[i],
                'end_frame': frames[i],
                'subtitle': current_sub
            }
    
    # 添加最後一個分組
    groups.append(current_group)
    
    return groups

# 將合併後的分組轉換為帶有時間戳的字幕格式
def generate_timestamped_subtitles(groups, fps):
    timestamped_subtitles = []
    for group in groups:
        start_time = frame_to_timestamp(group['start_frame'], fps)
        end_time = frame_to_timestamp(group['end_frame'], fps)
        subtitle = group['subtitle']
        timestamped_subtitles.append((start_time, end_time, subtitle))
    return timestamped_subtitles

# 將字幕輸出為 SRT 格式（可選）
def export_to_srt(timestamped_subtitles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, (start, end, subtitle) in enumerate(timestamped_subtitles, 1):
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle}\n\n")

# 主程式
def main():
    ocr_file = 'cleaned_ocr_output.txt'
    srt_output = 'output_subtitles.srt'  # 輸出 SRT 檔案路徑

    # 讀取 OCR 輸出
    frames, subtitles = read_ocr_output(ocr_file)
    print(f"共讀取到 {len(subtitles)} 幀的字幕。")

    # 合併相似字幕
    similarity_threshold = 0.4  # 相似度閾值
    groups = merge_subtitles(frames, subtitles, similarity_threshold)
    print(f"字幕已合併為 {len(groups)} 組。")

    # 生成帶有時間戳的字幕
    timestamped_subtitles = generate_timestamped_subtitles(groups, FPS)

    # 輸出為 SRT 格式
    export_to_srt(timestamped_subtitles, srt_output)
    print(f"SRT 字幕已輸出至 {srt_output}。")

    # 若需要在終端顯示部分結果，可以選擇顯示前幾個
    print("\n範例輸出（前 10 組）：")
    for ts in timestamped_subtitles[:10]:
        print(f"{ts[0]} --> {ts[1]}: {ts[2]}")

if __name__ == "__main__":
    main()
