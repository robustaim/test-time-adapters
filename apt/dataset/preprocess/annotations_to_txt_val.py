import os
import xml.etree.ElementTree as ET

# 데이터 경로 설정
annotations_root = "./apt/data/ILSVRC2015/Annotations/VID/val"  # XML 파일이 저장된 최상위 폴더
output_dir = "./apt/data/my_data/val/labels"  # YOLO 라벨 저장 폴더
os.makedirs(output_dir, exist_ok=True)

# WordNet Synset ID → YOLO class_id 변환
wnid_to_yolo = {
    "n02691156": 0, "n02419796": 1, "n02131653": 2, "n02834778": 3, "n01503061": 4,
    "n02924116": 5, "n02958343": 6, "n02402425": 7, "n02084071": 8, "n02121808": 9,
    "n02503517": 10, "n02118333": 11, "n02510455": 12, "n02342885": 13, "n02374451": 14,
    "n02129165": 15, "n01674464": 16, "n02484322": 17, "n03790512": 18, "n02324045": 19,
    "n02509815": 20, "n02411705": 21, "n01726692": 22, "n02355227": 23, "n02129604": 24,
    "n04468005": 25, "n01662784": 26, "n04530566": 27, "n02062744": 28, "n02391049": 29
}

# XML → YOLO 변환 함수
def convert_annotation(xml_path, txt_output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 이미지 크기 읽기 (width, height)
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    with open(txt_output_path, "w") as f:
        for obj in root.findall("object"):
            wnid = obj.find("name").text  # WordNet Synset ID 추출
            if wnid not in wnid_to_yolo:
                continue  # 매핑되지 않은 클래스는 스킵
            cls_id = wnid_to_yolo[wnid]  # YOLO 클래스 ID 변환

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # YOLO 형식 변환 (정규화)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 모든 하위 폴더에서 XML 파일 찾기 및 변환 실행
for root, _, files in os.walk(annotations_root):
    for xml_file in files:
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(root, xml_file)

            # 원본 폴더 구조 유지하며 labels 폴더에 저장
            relative_path = os.path.relpath(xml_path, annotations_root)
            txt_output_path = os.path.join(output_dir, relative_path.replace(".xml", ".txt"))
            
            # labels 폴더 안에 동일한 구조의 하위 폴더 생성
            os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

            convert_annotation(xml_path, txt_output_path)