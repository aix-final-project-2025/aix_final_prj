
#!/usr/bin/env python3
"""
batch_resize.py
의류 이미지 데이터셋을 224x224, 384x384 두 가지 해상도로
선명도를 최대한 유지하여 리사이즈하는 스크립트.

사용법:
    python batch_resize.py --src dataset/orig --dst dataset/resized
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm



def resize_and_save(img_path: Path, out_path: Path, size: int):
    """
    이미지 파일을 열고 정사각형(size x size)으로 리사이즈 후 저장.
    - LANCZOS 보간법 사용 → 디테일 보존
    - subsampling=0 → 색상 손실 최소화
    - quality=95 → 고화질 JPEG
    """
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((size, size), resample=Image.LANCZOS)  # 고품질 리사이즈
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, quality=95, subsampling=0)  # 고화질 저장
    except Exception as e:
        print(f"[ERROR] {img_path} 처리 실패 → {e}")


def process_dataset(src_root: Path, dst_root: Path, sizes: list[int]):
    """
    src_root 안의 모든 이미지를 각 사이즈별로 리사이즈하여 저장.
    """
    img_files = list(src_root.glob("*.*"))  # 모든 이미지 파일
    if not img_files:
        print(f"[WARN] {src_root} 안에 이미지 파일이 없습니다.")
        return

    for size in sizes:
        print(f"\n[INFO] {size}x{size} 리사이즈 시작")
        dst_size_dir = dst_root / str(size)
        dst_size_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(img_files, desc=f"Resizing {size}"):
            out_path = dst_size_dir / img_path.name
            resize_and_save(img_path, out_path, size)


def main():
    parser = argparse.ArgumentParser(description="이미지 데이터셋 일괄 리사이즈")
    parser.add_argument("--src", type=str, required=True, help="원본 데이터셋 폴더 경로")
    parser.add_argument("--dst", type=str, required=True, help="리사이즈된 데이터 저장 폴더")

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"원본 폴더가 없습니다: {src_root}")

    # 224, 384 두 가지 사이즈만 처리
    process_dataset(src_root, dst_root, sizes=[224, 384])
    print("\n✅ 모든 리사이즈 완료!")


if __name__ == "__main__":
    main()