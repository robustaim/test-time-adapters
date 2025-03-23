import os
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from functools import partial
import glob

# 경로 설정
label_source_root = "./apt/dataset/data/real_data/train/labels"
image_source_root = "./apt/dataset/data/ILSVRC2015/Data/VID/train"
output_dir = "./apt/dataset/data/real_data/train/img"
os.makedirs(output_dir, exist_ok=True)

def process_image(file, label_source_root, image_source_root, output_dir):
    label_path = os.path.join(label_source_root, file)

    # 원본 이미지 경로 생성
    parts = file.split("_")
    if len(parts) >= 3:
        vid_name = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3]
        mid_name = parts[4] + "_" + parts[5] + "_" + parts[6]
        image_name = parts[-1].replace(".txt", ".JPEG")
        src_image_path = os.path.join(image_source_root, vid_name, mid_name, image_name)
        dst_image_path = os.path.join(output_dir, file.replace(".txt", ".JPEG"))

        if os.path.exists(src_image_path):
            try:
                with Image.open(src_image_path) as img:
                    img_resized = img.resize((640, 640))
                    img_resized.save(dst_image_path, format="JPEG")
                return True
            except Exception as e:
                return f"오류: {src_image_path} - {str(e)}"
        else:
            return f"이미지 없음: {src_image_path}"
    return None

# 모든 라벨 파일 리스트 가져오기
txt_files = []
for root, _, files in os.walk(label_source_root):
    for file in files:
        if file.endswith(".txt"):
            # 상대 경로로 변환하여 저장
            rel_path = os.path.relpath(os.path.join(root, file), label_source_root)
            txt_files.append(rel_path)

# 병렬 처리
process_func = partial(process_image,
                       label_source_root=label_source_root,
                       image_source_root=image_source_root,
                       output_dir=output_dir)

# CPU 코어 수에 맞게 프로세스 풀 생성
num_workers = max(1, mp.cpu_count() - 1)
print(f"병렬 처리 시작 (프로세스 {num_workers}개 사용)")

with mp.Pool(num_workers) as pool:
    results = list(tqdm(pool.imap(process_func, txt_files), total=len(txt_files)))

# 오류 확인
errors = [r for r in results if isinstance(r, str)]
if errors:
    print(f"오류 발생 ({len(errors)} 건)")
    for e in errors[:10]:  # 처음 10개 오류만 출력
        print(e)

print(f"✅ 총 {len(txt_files)}개 파일 중 {len(results) - len(errors)}개 처리 완료!")
