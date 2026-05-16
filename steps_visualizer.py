import cv2
import os
from collections import Counter
from vision_pipeline import ANPRPipeline

# Configuration
IMAGE_PATH = "test_vehicle.jpg" 
OUTPUT_DIR = "presentation_images"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading ANPR Pipeline...")
    pipeline = ANPRPipeline()

    print(f"\nLoading image from: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print("Error: Could not load image. Check the IMAGE_PATH.")
        return

    # 1. Run YOLO Detection
    print("\n[STAGE 1] Running YOLO detection...")
    res = pipeline.detector.predict(img, conf=0.15, verbose=False)[0]
    
    if len(res.boxes) == 0:
        print("Error: No license plates detected in the image.")
        return

    boxes = res.boxes.data.cpu().numpy()
    best = max(boxes, key=lambda x: x[4])
    pad = 15
    x1, y1 = max(0, int(best[0])-pad), max(0, int(best[1])-pad)
    x2, y2 = min(img.shape[1], int(best[2])+pad), min(img.shape[0], int(best[3])+pad)

    # 2. Extract and Save Crops
    plate_crop = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_raw_yolo_crop.jpg"), plate_crop)
    
    crop_bgr = pipeline.ensure_3ch(plate_crop)
    h, w = crop_bgr.shape[:2]

    scale = 4
    up = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_upscaled_4x.jpg"), up)

    x_cut = int(0.22 * w)
    roi = crop_bgr[:, x_cut:] if (w - x_cut) > 20 else crop_bgr
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_roi_cut.jpg"), roi)

    variants = {
        "1_Upscaled": up,
        "2_Full_ROI": roi
    }

    rh, rw = roi.shape[:2]
    if rh > 20:
        top_half = roi[int(0.03*rh):int(0.48*rh), :]
        bottom_half = roi[int(0.45*rh):int(0.98*rh), :]
        variants["3_Top_Half"] = top_half
        variants["4_Bottom_Half"] = bottom_half
        cv2.imwrite(os.path.join(OUTPUT_DIR, "04_top_half_split.jpg"), top_half)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "05_bottom_half_split.jpg"), bottom_half)

    print("\n[STAGE 2 & 3] Processing Image Variants and OCR...")
    
    all_candidates = []
    
    # Loop through each variant and apply the 3 filtering stages
    for variant_name, img_variant in variants.items():
        print(f"\n>> Analyzing Variant: {variant_name}")
        
        # Define the 3 stages
        color_img = pipeline.ensure_3ch(img_variant)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_stage_A_color.jpg"), color_img)

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        sharp = pipeline._sharpen(pipeline._clahe(gray))
        sharp_3ch = pipeline.ensure_3ch(sharp)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_stage_B_sharpened.jpg"), sharp)

        _, th_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_3ch = pipeline.ensure_3ch(th_otsu)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_stage_C_otsu.jpg"), th_otsu)

        stages = {
            "Color": color_img,
            "Sharpened": sharp_3ch,
            "Otsu (B&W)": otsu_3ch
        }

        # Run OCR on all 3 stages for this specific variant
        for stage_name, stage_img in stages.items():
            results = pipeline.easy_reader.readtext(stage_img, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
            
            texts = [r[1] for r in results]
            scores = [r[2] for r in results]
            
            if not texts:
                print(f"  [{stage_name}] No text detected.")
                continue

            # Log the exact step-by-step transformation
            print(f"  [{stage_name}]")
            print(f"      1. Raw EasyOCR Read: {texts} (Confidence: {[round(s, 2) for s in scores]})")
            
            plate, mode = pipeline.extract_plate_from_pieces(texts, scores)
            if not plate or mode == "not_found":
                print("      2. Stitched Format: Failed (Did not match valid plate format)")
                continue
                
            print(f"      2. Stitched Format:  {plate} (Method: {mode})")
            
            corrected_plate = pipeline.apply_pakistani_corrections(plate)
            if corrected_plate != plate:
                print(f"      3. Matrix Applied:   {corrected_plate} <--- ERROR CORRECTED")
            else:
                print(f"      3. Matrix Applied:   {corrected_plate} (No fixes needed)")
                
            all_candidates.append(corrected_plate)

    # 4. Consensus Voting
    print("\n[STAGE 4] Consensus Voting")
    print("-" * 40)
    if not all_candidates:
        print("Final Result: FAILED (No valid candidates found across any variant)")
    else:
        vote_counts = Counter(all_candidates)
        print("Votes Tally:")
        for candidate, votes in vote_counts.items():
            print(f"  {candidate}: {votes} vote(s)")
            
        final_winner = vote_counts.most_common(1)[0][0]
        print(f"\n=> FINAL MAJORITY CONSENSUS: {final_winner}")
    print("-" * 40)
    print(f"\nSuccess! All presentation images have been saved to the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()