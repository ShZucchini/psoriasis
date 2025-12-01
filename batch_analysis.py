import cv2
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
from app import CSIFT_Algorithms  # Assuming your class is here

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

    # --- METRIC 1: REPEATABILITY (Geometric Check) ---
    # We project points from Img1 -> Img2 and check if they land near a point in Img2
    
    # Extract (x,y) coordinates
    pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
    pts2 = np.array([kp.pt for kp in kp2])
    
    # Transform points using the Rotation Matrix M
    # Note: cv2.transform is used for points, warpAffine for images
    pts1_projected = cv2.transform(pts1, M)
    
    correct_repeats = 0
    threshold = 3.0 # Allow 3 pixel error margin
    
    for pt_proj in pts1_projected:
        dest_x, dest_y = pt_proj[0]
        
        # Ignore points that rotated out of bounds
        if dest_x < 0 or dest_x >= w or dest_y < 0 or dest_y >= h:
            continue
            
        # Check distance to ANY keypoint in the second image
        # (Using simple Euclidean distance check)
        distances = np.linalg.norm(pts2 - np.array([dest_x, dest_y]), axis=1)
        if np.min(distances) < threshold:
            correct_repeats += 1
            
    # Repeatability = (Correct Re-detections / Total Original Points) * 100
    repeatability = (correct_repeats / len(kp1)) * 100

    # --- METRIC 2: MATCHING SCORE (Descriptor Check) ---
    # We check if the descriptors actually look similar
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    try:
        matches = bf.match(desc1, desc2)
        # Matching Score = (Valid Matches / Total Keypoints) * 100
        matching_score = (len(matches) / len(kp1)) * 100
    except:
        matching_score = 0.0

    return repeatability, matching_score

def run_batch_test(folder_path):
    results = []
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
            # START TIMER (SOP 1 metric)
            start = time.time()
            gray = CSIFT_Algorithms.rgb_to_invariant_iterative(img)
            sift = cv2.SIFT_create(contrastThreshold=0.04) # Standard Threshold
            k, d = sift.detectAndCompute(gray, None)
            # END TIMER
            t = (time.time() - start) * 1000 
            return k, d, t

        kp_std, desc_std, time_std = standard_wrapper(image)
        # Calculate Repeatability & Matching Score
        rep_std, match_std = calculate_metrics_and_matching(image, standard_wrapper)
        
        # --- 2. RUN ENHANCED ---
        def enhanced_wrapper(img):
            # START TIMER (SOP 1 metric)
            start = time.time()
            
            # Step A: Vectorized Conversion
            inv = CSIFT_Algorithms.rgb_to_invariant_vectorized(img)
            if len(inv.shape) == 3: inv = inv[:,:,0]
            
            # Step B: Texture Aware Detection (SOP 2)
            kp = CSIFT_Algorithms.texture_aware_detection(inv)
            
            # Step C: Descriptor & RootSIFT (SOP 3)
            sift = cv2.SIFT_create()
            _, desc = sift.compute(inv, kp)
            if desc is not None:
                desc = CSIFT_Algorithms.root_sift_normalization(desc)
                # Note: Ensure 'root_sift_normalization' also appends the color scalar!
            
            # END TIMER
            t = (time.time() - start) * 1000
            return kp, desc, t

        kp_enh, desc_enh, time_enh = enhanced_wrapper(image)
        rep_enh, match_enh = calculate_metrics_and_matching(image, enhanced_wrapper)
        
        # --- 3. LOG RESULTS ---
        results.append({
            "Filename": filename,
            
            # SOP 1: Efficiency
            "Std_Time_ms": round(time_std, 2),
            "Enh_Time_ms": round(time_enh, 2),
            
            # SOP 2: Detection Quality
            "Std_Keypoints": len(kp_std),
            "Enh_Keypoints": len(kp_enh),
            "Density_Increase": len(kp_enh) - len(kp_std),
            "Std_Repeatability": round(rep_std, 2),
            "Enh_Repeatability": round(rep_enh, 2),
            
            # SOP 3: Distinctiveness
            "Std_MatchingScore": round(match_std, 2),
            "Enh_MatchingScore": round(match_enh, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv("Thesis_Results_Final.csv", index=False)
    
    print("\n✅ Analysis Complete!")
    print(f"Avg Speed (ms): Standard={df['Std_Time_ms'].mean():.2f} vs Enhanced={df['Enh_Time_ms'].mean():.2f}")
    print(f"Avg Keypoint Increase: {df['Density_Increase'].mean():.2f}")
    print(f"Avg Repeatability: Std={df['Std_Repeatability'].mean():.2f}% vs Enh={df['Enh_Repeatability'].mean():.2f}%")
    print(f"Avg Matching Score: Std={df['Std_MatchingScore'].mean():.2f}% vs Enh={df['Enh_MatchingScore'].mean():.2f}%")

if __name__ == "__main__":
    run_batch_test("dataset")