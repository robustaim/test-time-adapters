import os
from PIL import Image

# 기존 이미지 및 라벨 저장 폴더
label_source_root = "/shared_hdd/annasdfghjkl13/APT/data/real_data/val/labels"
image_source_root = "/shared_hdd/annasdfghjkl13/APT/data/ILSVRC2015/Data/VID/val"
output_dir = "/shared_hdd/annasdfghjkl13/APT/data/real_data/val/img"  # 640x640 변환된 이미지 저장 폴더
os.makedirs(output_dir, exist_ok=True)

# 라벨 파일을 기반으로 해당하는 이미지 찾기
for root, _, files in os.walk(label_source_root):
    txt_files = sorted([f for f in files if f.endswith(".txt")])
    
    for file in txt_files:
        label_path = os.path.join(root, file)
        file_name = file.replace(".txt", ".JPEG")
        
        # 원본 이미지 경로 생성
        parts = file.split("_")
        if len(parts) >= 3:
            vid_name = parts[0] + "_" + parts[1] + "_" + parts[2]  # 예: ILSVRC2015_val_00000000
            image_name = parts[-1].replace(".txt", ".JPEG")
            src_image_path = os.path.join(image_source_root, vid_name, image_name)
            dst_image_path = os.path.join(output_dir, file.replace(".txt", ".JPEG"))
            
            if os.path.exists(src_image_path):
                with Image.open(src_image_path) as img:
                    img_resized = img.resize((640, 640))
                    img_resized.save(dst_image_path, format="JPEG")
            else:
                print(f"⚠️ 해당 이미지가 존재하지 않습니다: {src_image_path}")

print("✅ 라벨 파일을 기반으로 해당하는 이미지를 640x640으로 변환 후 저장 완료!")

