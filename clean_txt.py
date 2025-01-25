import re

# 讀取 OCR 輸出的文字檔案
def read_ocr_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# 清洗文字數據，保留行編號並僅保留中文字
def clean_ocr_text(file_path, output_path):
    raw_data = read_ocr_file(file_path)

    # 清洗數據並保留行編號與對應的中文字
    cleaned_data = []
    for line in raw_data:
        # 將行分為編號與文字部分
        parts = line.split('\t', 1)
        if len(parts) == 2:
            line_id, text = parts
            # 僅保留中文字
            cleaned_text = re.sub(r"[^\u4e00-\u9fff]", "", text)
            cleaned_data.append(f"{line_id}\t{cleaned_text}\n")

    # 將清洗後的數據寫回新檔案
    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_data)

# 主程序
input_file = "ocr_output.txt"  # 替換為您的原始檔案路徑
output_file = "cleaned_ocr_output.txt"  # 清洗後的檔案路徑

clean_ocr_text(input_file, output_file)

print("文字清洗完成，結果已保存至 cleaned_ocr_output.txt")
