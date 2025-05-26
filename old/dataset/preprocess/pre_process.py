#for imagenetVID
class pre_process:
    def __init__(self, annotation_root, img_root, output_root):
        self.annotation_root = annotation_root
        self.img_root = img_root
        self.output_root = output_root
        
    def convert_annotation(xml_path, output_root): #annotation to txt
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 이미지 크기 읽기 (width, height)
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        with open(output_root, "w") as f:
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