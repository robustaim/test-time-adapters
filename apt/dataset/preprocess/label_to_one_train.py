import os
import shutil

# 기존 라벨 루트 폴더
label_source_root = "./apt/dataset/data/my_data/train/labels"

# 새로운 라벨 저장 폴더
label_output_dir = "./apt/dataset/data/real_data/train/labels"
os.makedirs(label_output_dir, exist_ok=True)

# 모든 하위 폴더에서 `.txt` 파일을 찾아 빈 파일 제외 후 처리
for root, _, files in os.walk(label_source_root):
    txt_files = sorted([f for f in files if f.endswith(".txt")])
    
    valid_files = []
    for file in txt_files:
        src_txt_path = os.path.join(root, file)
        
        # 빈 파일이 아닌 경우만 리스트에 추가
        if os.path.getsize(src_txt_path) > 0:
            valid_files.append(file)
    
    # 전체 파일 선택(빈 파일 제외)
    for file in valid_files:
        src_txt_path = os.path.join(root, file)
        
        parts = root.split(os.sep)
        if len(parts) >= 3:
            vid_name = parts[-2]  # ILSVRC2015_VID_train_0000
            train_name = parts[-1]  # ILSVRC2015_train_00000000
            
            new_filename = f"{vid_name}_{train_name}_{file}"
            dst_txt_path = os.path.join(label_output_dir, new_filename)
            
            shutil.copy(src_txt_path, dst_txt_path)  # 원본을 유지하면서 복사

print("✅ 유효한 라벨을 가진 상위 5개의 라벨을 복사 완료!")
