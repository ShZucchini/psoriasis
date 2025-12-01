import cv2
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
from app import CSIFT_Algorithms

def calculate_repeatability_metrics(image, detector_func):
    """
    Calculates Repeatability AND Distinctiveness (Variance).
    """
    # 1. Run Algorithm on Original
    kp1, desc1, _ = detector_func(image)
    
    # Validation: If no features found, return 0
    if desc1 is None or len(kp1) < 2: 
        return 0.0, 0.0
    
    # Calculate Distinctiveness (Variance of descriptors)
    distinctiveness = np.var(desc1)

    # 2. Rotate Image 15 degrees for Repeatability Test
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    
    # 3. Run Algorithm on Rotated
    kp2, desc2, _ = detector_func(rotated_img)
    if desc2 is None or len(kp2) < 2: 
        return 0.0, distinctiveness

    # 4. Match them (Brute Force Matcher)
    # Ensure float32 for OpenCV Matcher compatibility
    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    try:
        matches = bf.match(desc1, desc2)
        # Repeatability = (Matches / Min Keypoints) * 100
        min_kp = min(len(kp1), len(kp2))
        repeatability = (len(matches) / min_kp) * 100 if min_kp > 0 else 0
        return repeatability, distinctiveness
    except Exception as e:
        return 0.0, distinctiveness

def run_batch_test(folder_path):
    results = []
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📂 Found {len(files)} images. Starting standardized analysis...")
    
    for filename in tqdm(files):
        img_path = os.path.join(folder_path, filename)
        image_original = cv2.imread(img_path)
        if image_original is None: continue
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        
        # --- CRITICAL FIX: PRE-PROCESSING (Standardize Resolution) ---
        # Resize both to Width 600px (Scientific Control Variable)
        target_width = 600
        h, w, c = image_original.shape
        scale = target_width / w
        new_h = int(h * scale)
        image = cv2.resize(image_original, (target_width, new_h))
        
        # --- 1. RUN STANDARD ---
        def standard_wrapper(img):
            start = time.time()
            # No internal resizing here anymore. Processing the exact 600px image.
            gray = CSIFT_Algorithms.rgb_to_invariant_iterative(img)
            
            sift = cv2.SIFT_create(contrastThreshold=0.04)
            k, d = sift.detectAndCompute(gray, None)
            t = (time.time() - start) * 1000
            return k, d, t

        kp_std, desc_std, time_std = standard_wrapper(image)
        rep_std, dist_std = calculate_repeatability_metrics(image, standard_wrapper)
        
        # --- 2. RUN ENHANCED ---
        def enhanced_wrapper(img):
            start = time.time()
            # Processing the exact same 600px image.
            inv = CSIFT_Algorithms.rgb_to_invariant_vectorized(img)
            if len(inv.shape) == 3: inv = inv[:,:,0]
            kp = CSIFT_Algorithms.texture_aware_detection(inv)
            sift = cv2.SIFT_create()
            _, desc = sift.compute(inv, kp)
            desc = CSIFT_Algorithms.root_sift_normalization(desc)
            t = (time.time() - start) * 1000
            return kp, desc, t

        kp_enh, desc_enh, time_enh = enhanced_wrapper(image)
        rep_enh, dist_enh = calculate_repeatability_metrics(image, enhanced_wrapper)
        
        # --- 3. LOG RESULTS ---
        results.append({
            "Filename": filename,
            "Std_Time_ms": round(time_std, 2),
            "Enh_Time_ms": round(time_enh, 2),
            "Time_Improvement_%": round(((time_std - time_enh) / time_std) * 100, 2),
            "Std_Keypoints": len(kp_std),
            "Enh_Keypoints": len(kp_enh),
            "Density_Increase": len(kp_enh) - len(kp_std),
            "Std_Repeatability": round(rep_std, 2),
            "Enh_Repeatability": round(rep_enh, 2),
            "Std_Distinctiveness": round(dist_std, 2),
            "Enh_Distinctiveness": round(dist_enh, 2)
        })

    df = pd.DataFrame(results)
    # Saving to V3 to differentiate
    df.to_csv("Thesis_Results_Summary_V3.csv", index=False)
    
    print("\n✅ Analysis Complete!")
    print(f"Avg Speed Improvement: {df['Time_Improvement_%'].mean():.2f}%")
    print(f"Avg Keypoint Increase: {df['Density_Increase'].mean():.2f}")
    print(f"Avg Standard Repeatability: {df['Std_Repeatability'].mean():.2f}%")
    print(f"Avg Enhanced Repeatability: {df['Enh_Repeatability'].mean():.2f}%")

if __name__ == "__main__":
    run_batch_test("dataset")