import cv2
from PIL import Image
import os

def extract_frames(video_path, output_dir):
    """
    從影片中提取關鍵幀並儲存為圖像檔案。

    :param video_path: 影片檔案路徑
    :param output_dir: 輸出圖像的目錄
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 嘗試打開影片檔案
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法打開影片: {video_path}")
        return

    # 取得影片的總幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片總幀數: {total_frames}")

    # 計算需要提取的幀索引
    key_frame_indices = [1, total_frames // 2, total_frames - 2]

    # 提取並儲存指定幀
    for frame_index, frame_position in enumerate(key_frame_indices, start=1):
        save_frame_at_position(cap, frame_position, output_dir, frame_index)

    # 釋放資源
    cap.release()
    print("影片處理完成。")

def save_frame_at_position(cap, position, output_dir, frame_index):
    """
    儲存影片中的指定幀為圖像檔案。

    :param cap: cv2.VideoCapture 物件
    :param position: 幀的位置（索引）
    :param output_dir: 圖像輸出目錄
    :param frame_index: 幀的編號（儲存檔案時使用）
    """
    # 設定幀位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    
    # 讀取指定幀
    ret, frame = cap.read()
    if not ret:
        print(f"無法讀取幀: {position}")
        return
    
    # 儲存幀圖像至輸出目錄
    output_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"成功儲存幀至: {output_path}")

def show_image_info(image_path):
    """
    顯示圖片資訊（大小與色彩模式）。

    :param image_path: 圖片檔案路徑
    """
    image = Image.open(image_path).convert('RGB')
    print(f"圖像尺寸: {image.size}")

# 使用範例
if __name__ == "__main__":
    # 輸入的影片路徑
    video_path = 'example.mp4'
    
    # 圖像輸出目錄
    output_dir = "./data"
    
    # 提取幀並儲存
    extract_frames(video_path, output_dir)
    
    # 顯示提取的第一幀圖片資訊
    first_frame_path = os.path.join(output_dir, 'frame_1.jpg')
    if os.path.exists(first_frame_path):
        show_image_info(first_frame_path)
    else:
        print("第一幀圖像未成功儲存，無法顯示資訊。")
