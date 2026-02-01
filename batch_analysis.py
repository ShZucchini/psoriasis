import cv2
import numpy as np
import os
import pandas as pd
import time
import tracemalloc
import gc
from tqdm import tqdm
from app import CSIFT_Algorithms

def run_batch_test(folder_path):
    results = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📂 Found {len(files)} images. Starting Comparative Analysis...")
    
    for filename in tqdm(files):
        img_path = os.path.join(folder_path, filename)
        image_raw = cv2.imread(img_path)
        if image_raw is None: continue
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        
        # Standardize Resolution (Matching app.py logic)
        target_width = 600
        h, w, _ = image_raw.shape
        scale = target_width / w
        image = cv2.resize(image_raw, (target_width, int(h * scale)))

        # --- 1. DEFINE DETECTOR WRAPPERS (Mirrors app.py) ---

        def run_sift(img):
            gc.collect()
            tracemalloc.start()
            start = time.time()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sift = cv2.SIFT_create()
            k, d = sift.detectAndCompute(gray, None)
            t = (time.time() - start) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return k, d, t, (peak / (1024 * 1024))
        
        def run_standard(img):
            gc.collect()
            tracemalloc.start()
            start = time.time()
            # SOP 1: Iterative Grayscale
            gray = CSIFT_Algorithms.rgb_to_invariant_iterative(img)
            sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.04)
            k, d = sift.detectAndCompute(gray, None)
            t = (time.time() - start) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return k, d, t, (peak / (1024 * 1024))

        def run_enhanced(img):
            gc.collect()
            tracemalloc.start()
            start = time.time()
            # SOP 1 & 2: Vectorized + Adaptive Masking
            inv = CSIFT_Algorithms.rgb_to_invariant_vectorized(img)
            kp = CSIFT_Algorithms.texture_aware_detection(inv, img)
            # SOP 3: RootSIFT
            sift = cv2.SIFT_create()
            _, desc = sift.compute(inv, kp)
            desc_root = CSIFT_Algorithms.root_sift_normalization(desc)
            t = (time.time() - start) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return kp, desc_root, t, (peak / (1024 * 1024))

        # --- 2. EXECUTE AND CALCULATE METRICS ---
        kp_sift, ds_sift, t_sift, m_sift = run_sift(image)
        kp_std, ds_std, t_std, m_std = run_standard(image)
        kp_enh, ds_enh, t_enh, m_enh = run_enhanced(image)

        # Import the exact metric calculator from your app.py logic
        from app import calculate_real_metrics_live
        rep_sift, match_sift = calculate_real_metrics_live(image, kp_sift, ds_sift, run_sift)
        rep_std, match_std = calculate_real_metrics_live(image, kp_std, ds_std, run_standard)
        rep_enh, match_enh = calculate_real_metrics_live(image, kp_enh, ds_enh, run_enhanced)
        
        results.append({
            "Filename": filename,
            "SIFT_Time_ms": round(t_sift, 2), "Std_Time_ms": round(t_std, 2), "Enh_Time_ms": round(t_enh, 2),
            "SIFT_Mem_MB": round(m_sift, 2), "Std_Mem_MB": round(m_std, 2), "Enh_Mem_MB": round(m_enh, 2),
            "SIFT_Density": len(kp_sift), "Std_Density": len(kp_std), "Enh_Density": len(kp_enh),
            "SIFT_Rep_%": round(rep_sift, 2), "Std_Rep_%": round(rep_std, 2), "Enh_Rep_%": round(rep_enh, 2),
            "SIFT_Match_%": round(match_sift, 2), "Std_Match_%": round(match_std, 2), "Enh_Match_%": round(match_enh, 2)
        })

    # --- 3. FINAL SUMMARY TABLEv8
    if results:
        df = pd.DataFrame(results)
        df.to_csv("ThreeAlgo_Batch_Results.csv", index=False)
        
        print(f"\n✅ Analysis Complete! Results saved to 'ThreeAlgo_Batch_Results.csv'")
        print("="*85)
        print(f"{'METRIC':<25} | {'SIFT':<15} | {'STANDARD':<15} | {'ENHANCED':<15}")
        print("-"*85)
        metrics = [
            ('Avg Time (ms)', 'Time_ms'),
            ('Avg Memory (MB)', 'Mem_MB'),
            ('Avg Density', 'Density'),
            ('Avg Repeatability %', 'Rep_%'),
            ('Avg Matching Score %', 'Match_%')
        ]
        for label, key in metrics:
            print(f"{label:<25} | {df['SIFT_'+key].mean():>12.2f}  | {df['Std_'+key].mean():>12.2f}  | {df['Enh_'+key].mean():>12.2f}")
        print("="*85)

if __name__ == "__main__":
    run_batch_test("newdataset")