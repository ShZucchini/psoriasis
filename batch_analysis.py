import cv2
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
from app import CSIFT_Algorithms

def calculate_metrics_and_matching(image, detector_func):
    """
    Calculates Repeatability (Geometric check) AND Matching Score (Descriptor check).
    """
    # 1. Run Algorithm on Original Image
    kp1, desc1, _ = detector_func(image)
    if desc1 is None or len(kp1) < 2: return 0.0, 0.0
    
    # 2. Create Synthetic Rotation (15 degrees)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0) # Rotation Matrix
    rotated_img = cv2.warpAffine(image, M, (w, h))
    
    # 3. Run Algorithm on Rotated Image
    kp2, desc2, _ = detector_func(rotated_img)
    if desc2 is None or len(kp2) < 2: return 0.0, 0.0

    # --- METRIC 1: REPEATABILITY ---
    pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
    pts2 = np.array([kp.pt for kp in kp2])
    
    # Transform points using the Rotation Matrix M
    pts1_projected = cv2.transform(pts1, M)
    
    correct_repeats = 0
    threshold = 5.0 # Increased slightly to 5.0 to match app.py strictness
    
    for pt_proj in pts1_projected:
        dest_x, dest_y = pt_proj[0]
        
        if dest_x < 0 or dest_x >= w or dest_y < 0 or dest_y >= h:
            continue
            
        distances = np.linalg.norm(pts2 - np.array([dest_x, dest_y]), axis=1)
        if np.min(distances) < threshold:
            correct_repeats += 1
            
    repeatability = (correct_repeats / len(kp1)) * 100

    # --- METRIC 2: MATCHING SCORE ---
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    try:
        matches = bf.match(desc1, desc2)
        matching_score = (len(matches) / len(kp1)) * 100
    except:
        matching_score = 0.0

    return repeatability, matching_score

def run_batch_test(folder_path):
    results = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📂 Found {len(files)} images. Starting standardized analysis...")
    
    for filename in tqdm(files):
        img_path = os.path.join(folder_path, filename)
        image_original = cv2.imread(img_path)
        if image_original is None: continue
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        
        # --- PRE-PROCESSING (Standardize Resolution) ---
        target_width = 600
        h, w, c = image_original.shape
        scale = target_width / w
        new_h = int(h * scale)
        image = cv2.resize(image_original, (target_width, new_h))
        
        # --- 1. RUN STANDARD ---
        def standard_wrapper(img):
            start = time.time()
            # Note: Standard uses Iterative gray conversion
            gray = CSIFT_Algorithms.rgb_to_invariant_iterative(img)
            sift = cv2.SIFT_create(contrastThreshold=0.04) 
            k, d = sift.detectAndCompute(gray, None)
            t = (time.time() - start) * 1000 
            return k, d, t

        kp_std, desc_std, time_std = standard_wrapper(image)
        rep_std, match_std = calculate_metrics_and_matching(image, standard_wrapper)
        
        # --- 2. RUN ENHANCED ---
        def enhanced_wrapper(img):
            start = time.time()
            
            # Step A: Preprocess (Hair Removal/Denoise) - CRITICAL UPDATE
            clean_image = CSIFT_Algorithms.preprocess_image(img)
            
            # Step B: Vectorized Conversion
            inv = CSIFT_Algorithms.rgb_to_invariant_vectorized(clean_image)
            if len(inv.shape) == 3: inv = inv[:,:,0]
            
            # Step C: Texture Aware Detection (Pass TWO arguments now)
            kp = CSIFT_Algorithms.texture_aware_detection(inv, clean_image)
            
            # Step D: Descriptor & RootSIFT
            sift = cv2.SIFT_create()
            _, desc = sift.compute(inv, kp)
            
            # RootSIFT Normalization
            if desc is not None:
                desc = CSIFT_Algorithms.root_sift_normalization(desc)

            t = (time.time() - start) * 1000
            return kp, desc, t

        kp_enh, desc_enh, time_enh = enhanced_wrapper(image)
        rep_enh, match_enh = calculate_metrics_and_matching(image, enhanced_wrapper)
        
        # --- 3. LOG RESULTS ---
        results.append({
            "Filename": filename,
            "Std_Time_ms": round(time_std, 2),
            "Enh_Time_ms": round(time_enh, 2),
            "Std_Keypoints": len(kp_std),
            "Enh_Keypoints": len(kp_enh),
            "Density_Increase": len(kp_enh) - len(kp_std),
            "Std_Repeatability": round(rep_std, 2),
            "Enh_Repeatability": round(rep_enh, 2),
            "Std_MatchingScore": round(match_std, 2),
            "Enh_MatchingScore": round(match_enh, 2)
        })

    if not results:
        print("❌ No valid images processed.")
        return

    df = pd.DataFrame(results)
    output_file = "Thesis_Results_Final.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Analysis Complete! Saved to {output_file}")
    print(f"Avg Speed (ms): Standard={df['Std_Time_ms'].mean():.2f} vs Enhanced={df['Enh_Time_ms'].mean():.2f}")
    print(f"Avg Keypoint Increase: {df['Density_Increase'].mean():.2f}")
    print(f"Avg Repeatability: Std={df['Std_Repeatability'].mean():.2f}% vs Enh={df['Enh_Repeatability'].mean():.2f}%")
    print(f"Avg Matching Score: Std={df['Std_MatchingScore'].mean():.2f}% vs Enh={df['Enh_MatchingScore'].mean():.2f}%")

if __name__ == "__main__":
    # Make sure you have a folder named 'dataset' with images inside
    run_batch_test("dataset")