# 此檔案請放在Tinyllava_factory裡面的路徑使用，並且記得看word增加程式碼，影片路徑目前是寫死所以要記得更改
from tinyllava.eval.run_tiny_llava import eval_model_tv , ocr_video  
import numpy as np
import Levenshtein
import os
# model_path = "/home/kevin/TinyLLaVA_Factory/tiny-llava-kevin_llama3_taide-siglip-so400m-patch14-384-base-lora-zero2-r128-finetune"
model_path ='/home/kevin/TinyLLaVA_Factory/tiny-llava-taigi-alpaca-v1-siglip-so400m-patch14-384-base-lora-zero2-r128-finetune'
prompt = "Please identify the subtitles below the picture\n"
image_file = f'/nfs/RS2416RP/Workspace/spec/synth/test0312/test_image_5492.jpg'
conv_mode = "llama" # or llama, gemma, etc
args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 30 
})()
ocr_video(args)