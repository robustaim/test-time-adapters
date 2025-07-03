import cv2
import os
from tqdm import tqdm

'''
이 파일 실행 시 기존 val dataset이랑 동일한 파일 구조로 파일이 추가됨
val -> val_c
나머지는 동일
'''
n = 5 # n개까지는 원본
def convert(image): 
    # 뿌연 막 만들기: 대비 감소와 밝기 증가
    alpha = 0.5  # 대비 감소 계수 (0에 가까울수록 뿌옇게)
    beta = 100   # 밝기 증가 (0~255)

    # 대비와 밝기 조절
    noise_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return noise_image

def count_files(input_root_dir):  # 전체 파일 갯수
    total_count = 0
    for root, dirs, files in os.walk(input_root_dir):
        for file in files:
            if file.endswith(".JPEG"):
                total_count += 1
    return total_count

def apply_noise_to_dataset(input_root_dir, output_root_dir):
    total_files = count_files(input_root_dir)
    print(f"총 이미지 수: {total_files}")

    # 데이터셋 폴더 순회
    with tqdm(total=total_files, desc="노이즈 추가 중", unit="파일") as pbar:
        for root, dirs, files in os.walk(input_root_dir):
            # 파일 이름을 정렬하여 처리 순서 보장
            files = sorted(files)

            for idx, file in enumerate(files): 
                if file.endswith(".JPEG"): # root: ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000 | files: 000000.JPEG
                    # 처음 10개의 프레임은 노이즈 추가하지 않음
                    if idx < n: # n개 까지는 원본 유지지
                        # 입력 파일 경로
                        file_path = os.path.join(root, file)

                        # 이미지 로드
                        image = cv2.imread(file_path)
                        if image is None:
                            print(f"이미지 로드 실패: {file_path}")
                            continue
                        
                        # 상대 경로 계산하여 출력 경로 생성
                        relative_path = os.path.relpath(root, input_root_dir) # relative_path: ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000
                        output_dir = os.path.join(output_root_dir, relative_path)
                        
                        # 폴더가 없으면 생성
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 원본 이미지 그대로 저장
                        output_path = os.path.join(output_dir, file)
                        cv2.imwrite(output_path, image)
                        pbar.update(1)
                        continue

                    # 이후 프레임은 노이즈 추가
                    file_path = os.path.join(root, file)

                    # 이미지 로드
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"이미지 로드 실패: {file_path}")
                        continue
                    
                    # 노이즈 입히기
                    noise_image = convert(image)
                    
                    # 상대 경로 계산하여 출력 경로 생성
                    relative_path = os.path.relpath(root, input_root_dir)
                    output_dir = os.path.join(output_root_dir, relative_path)
                    
                    # 폴더가 없으면 생성
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 변환된 이미지 저장
                    output_path = os.path.join(output_dir, file)
                    cv2.imwrite(output_path, noise_image)
                    
                    # 진행 바 업데이트
                    pbar.update(1)

# 데이터셋 경로 설정
input_dataset_path = "../data/ILSVRC2015/Data/VID/val"
output_dataset_path = "../data/ILSVRC2015/Data/VID/val_c"

apply_noise_to_dataset(input_dataset_path, output_dataset_path)

'''
└── Data
    └── VID
        └── val
            ├── ILSVRC2015_VID_train_0000
            │    ├── ILSVRC2015_train_00000000
            │    │    ├── 000000.JPEG
            │    │    ├── 000001.JPEG
            │    │    └── ...
            │    ├── ILSVRC2015_train_00000001
            │    │    ├── 000000.JPEG
            │    │    ├── 000001.JPEG
            │    │    └── ...
            │    └── ...
            ├── ILSVRC2015_VID_train_0001
            │    ├── ILSVRC2015_train_00000000
            │    │    ├── 000000.JPEG
            │    │    ├── 000001.JPEG
            │    │    └── ...
            │    └── ...
            └── ... 
'''
