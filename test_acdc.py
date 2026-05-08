import sys
sys.path.insert(0, '.')

from ttadapters.datasets.acdc import ACDCDataset, ACDCDatasetForObjectDetection

ROOT = "./data"

# Step 1: annotations + images 다운로드
print("=== Downloading detection annotations + images ===")
ACDCDataset.download(
    root=f"{ROOT}/ACDC",
    force=False,
    download_key=("detection", "images")
)

# Step 2: 정상 인스턴스화
print("\n=== Instantiating dataset (val/fog) ===")
ds = ACDCDatasetForObjectDetection(
    root=ROOT,
    train=True,
    valid=True,
    corruption_type=ACDCDatasetForObjectDetection.CorruptionType.FOG,
    force_download=False,
)

print(f"len(ds): {len(ds)}")

# Step 3: __getitem__ 검증
print("\n=== Testing __getitem__ ===")
image, target = ds[0]
print(f"image shape  : {image.shape}")
print(f"boxes2d      : {target['boxes2d'][:2]}")
print(f"boxes2d_classes: {target['boxes2d_classes'][:2]}")
print(f"original_hw  : {target['original_hw']}")
