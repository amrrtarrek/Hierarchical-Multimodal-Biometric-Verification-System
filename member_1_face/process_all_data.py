"""
process_all_data.py
Robust batch processing pipeline for all biometric subjects.
"""
import os
import sys
import cv2
from src.face_preprocessing import preprocess_face_image, verify_dataset

FACE_SIZE = (100, 100)
VALID_EXTS = (".jpg", ".jpeg", ".png", ".pgm", ".bmp", ".webp")


def find_existing_dir(candidates: list) -> str | None:
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def process_directory(raw_dir: str, save_dir: str, prefix: str) -> int:
    if not os.path.isdir(raw_dir):
        return 0

    files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(VALID_EXTS)])
    if not files:
        print(f"  [WARNING] Directory '{raw_dir}' exists but contains no image files.")
        return 0

    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    for f in files:
        path = os.path.join(raw_dir, f)
        img = cv2.imread(path)
        if img is None:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if img is None:
            continue

        processed = preprocess_face_image(img)
        if processed is None:
            # Fallback for pre-cropped face datasets (e.g. AT&T ORL)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed = cv2.resize(cv2.equalizeHist(gray), FACE_SIZE, interpolation=cv2.INTER_AREA)

        out_name = f"{prefix}_{saved:03d}.png"
        cv2.imwrite(os.path.join(save_dir, out_name), processed)
        saved += 1
    return saved


def main():
    print("=" * 60)
    print("  TASK 1.1: BATCH FACE PREPROCESSING & CONDITIONING")
    print("=" * 60)

    # 1. Member 1 (Class 1 - You)
    m1_candidates = ["data/raw/member-1", "data/raw/member_1", "data/raw/member1"]
    m1_dir = find_existing_dir(m1_candidates)
    if m1_dir:
        n1 = process_directory(m1_dir, "data/processed/subject_001", "img")
        print(f"  [OK] subject_001 (Member 1 - You): {n1} images processed from '{m1_dir}'")
        if n1 == 0:
            print("       -> TIP: Run 'python capture_laptop_webcam.py' to capture your 20 photos!")
    else:
        print(f"  [MISSING] No Member 1 raw folder found. Checked: {m1_candidates}")
        print("            -> Run 'python capture_laptop_webcam.py' to capture your photos!")

    # 2. Member 2 (Class 2 - Enrolled)
    m2_candidates = ["data/raw/member-2", "data/raw/member_2", "data/raw/member2"]
    m2_dir = find_existing_dir(m2_candidates)
    if m2_dir:
        n2 = process_directory(m2_dir, "data/processed/subject_002", "img")
        print(f"  [OK] subject_002 (Member 2): {n2} images processed from '{m2_dir}'")
    else:
        print(f"  [INFO] Member 2 folder not found (optional in standalone test).")

    # 3. AT&T Dataset (Classes 3-42)
    att_candidates = ["data/raw/att-faces", "data/raw/att_faces", "data/raw/att_face", "data/raw/ORL"]
    att_dir = find_existing_dir(att_candidates)
    if att_dir:
        att_count = 0
        for s in range(1, 41):
            s_dir = os.path.join(att_dir, f"s{s}")
            if os.path.isdir(s_dir):
                k = process_directory(s_dir, f"data/processed/subject_{s+2:03d}", "att")
                if k > 0:
                    att_count += 1
        print(f"  [OK] AT&T subjects (003-042): {att_count} subject folders processed from '{att_dir}'")
    else:
        print(f"  [MISSING] AT&T raw dataset folder not found. Checked: {att_candidates}")

    # 4. Impostors (Class 0)
    imp_candidates = [
        "data/raw/impostors/member-3", "data/raw/impostors/member-4",
        "data/raw/impostors", "data/raw/imposter", "data/raw/imposters"
    ]
    imp_saved = 0
    for d in imp_candidates:
        if os.path.isdir(d):
            files = sorted([f for f in os.listdir(d) if f.lower().endswith(VALID_EXTS)])
            for f in files:
                p = os.path.join(d, f)
                if not os.path.isfile(p):
                    continue
                img = cv2.imread(p)
                if img is None:
                    continue
                crop = preprocess_face_image(img)
                if crop is not None:
                    os.makedirs("data/processed/subject_000", exist_ok=True)
                    cv2.imwrite(f"data/processed/subject_000/imp_{imp_saved:03d}.png", crop)
                    imp_saved += 1
    if imp_saved > 0:
        print(f"  [OK] subject_000 (Class 0 Impostors): {imp_saved} images processed.")
    else:
        print(f"  [WARNING] No impostor images processed into subject_000.")

    # 5. Clean up any empty subject folders created by accident
    if os.path.isdir("data/processed"):
        for s in os.listdir("data/processed"):
            sp = os.path.join("data/processed", s)
            if os.path.isdir(sp) and len(os.listdir(sp)) == 0:
                os.rmdir(sp)

    # 6. Verify Definition of Done
    if not os.path.isdir("data/processed") or len(os.listdir("data/processed")) == 0:
        print("\n[ERROR] No processed subjects found in 'data/processed'.")
        print("Please ensure your raw photos are placed in 'data/raw/' before running this script.")
        sys.exit(1)

    ok = verify_dataset("data/processed")
    if ok:
        print("\n[SUCCESS] All face crops verified as 100x100 uint8 arrays.")
    else:
        print("\n[FAILED] Dataset verification failed! Check image dimensions in data/processed/.")
        sys.exit(1)


if __name__ == "__main__":
    main()
