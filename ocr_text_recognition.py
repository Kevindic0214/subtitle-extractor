from paddleocr import PaddleOCR
import time

# 選擇模型
# 如果使用 fine-tuned 模型，取消註解以下兩行
# rec_model_dir = './inference/_cht_PP-OCRv3_rec/'
# ocr = PaddleOCR(use_angle_cls=False, lang='chinese_cht', use_gpu=True, rec_model_dir=rec_model_dir)

# 如果使用 pre-trained 模型，請保留以下初始化
ocr = PaddleOCR(use_angle_cls=True, lang='chinese_cht', use_gpu=0)

# 設定圖片路徑
# 可以根據需要替換成要辨識的圖片檔案
# img_path = '/home/kevin/TinyLLaVA_Factory/data/frame_1.jpg'
img_path = '/home/kevin/TinyLLaVA_Factory/data/frame_2.jpg'

# 計時開始
start_time = time.time()

# 執行 OCR 辨識
result = ocr.ocr(img_path, cls=False)

# 計算並打印執行耗時
elapsed_time = time.time() - start_time
print(f"OCR 執行時間: {elapsed_time:.2f} 秒")

# 處理與打印結果
if result[0] is not None:  # 確保有有效的辨識結果
    for block_idx, block in enumerate(result):  # 遍歷每個文字區塊
        print(f"\n區塊 {block_idx + 1}:")
        for line in block:  # 遍歷區塊內每一行的文字
            position, (text, confidence) = line[0], line[1]
            print(f"位置: {position}, 辨識文字: '{text}', 置信度: {confidence:.2f}")
else:
    print("未識別到文字。")


# 以下為使用 CnOcr 的代碼範例
"""
from cnocr import CnOcr

圖片路徑
img_fp = '/home/kevin/TinyLLaVA_Factory/data/image/image_1.jpg'
img_fp = '/nfs/RS2416RP/Workspace/spec/TPTS/picture/大港的台灣/lcw8BJhhK7M_230066_232432.jpg'
img_fp = "/home/kevin/TinyLLaVA_Factory/frame_5.jpg" 
img_fp = '/home/kevin/PaddleOCR/show_img.jpg'

計時開始
n = time.time()

初始化 CnOcr
ocr = CnOcr(rec_model_name='chinese_cht_PP-OCRv3')  # 識別模型使用繁體識別模型

執行 OCR
out = ocr.ocr(img_fp, cls=True)

打印耗時
print(time.time() - n)

打印結果
print(out)
"""