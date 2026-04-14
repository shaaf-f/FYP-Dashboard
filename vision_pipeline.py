import cv2
import re
import numpy as np
from collections import Counter
from ultralytics import YOLO
import easyocr

class ANPRPipeline:
    def __init__(self, weights_path="./Weights/best.pt", use_gpu=False):
        print("Initializing ANPR Pipeline...")
        self.detector = YOLO(weights_path)
        if use_gpu:
            self.detector.to('cuda')
        self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("ANPR Pipeline Ready.")

    def ensure_3ch(self, img):
        if img is None: return None
        if len(img.shape) == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(img.shape) == 3 and img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _clahe(self, gray):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return c.apply(gray)

    def _sharpen(self, gray):
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, k)

    def apply_pakistani_corrections(self, plate_str):
        if not plate_str or "-" not in plate_str: return plate_str
        prefix, suffix = plate_str.split("-", 1)
        to_alpha = str.maketrans("01258", "OIZSB")
        to_digit = str.maketrans("OIZSB", "01258")
        
        new_prefix = re.sub(r'[^A-Z]', '', prefix.translate(to_alpha))
        new_suffix = re.sub(r'[^0-9]', '', suffix.translate(to_digit))
        
        if not new_prefix or not new_suffix: return plate_str
        return f"{new_prefix}-{new_suffix}"

    def extract_plate_from_pieces(self, texts, scores):
        if not texts: return "", "empty"
        pieces = [(re.sub(r'[^A-Z0-9\-\s]', '', str(t).upper()).strip(), s) for t, s in zip(texts, scores) if re.sub(r'[^A-Z0-9\-\s]', '', str(t).upper()).strip()]
        if not pieces: return "", "empty"

        full_candidates = []
        for t, s in pieces:
            compact = re.sub(r'[\s\-]+', '', t)
            m = re.fullmatch(r'([A-Z]{2,3})(\d{3,4})', compact)
            if m: full_candidates.append((f"{m.group(1)}-{m.group(2)}", s))
        if full_candidates: return max(full_candidates, key=lambda x: x[1])[0], "full_token"

        letters, d4, d3 = [], [], []
        for t, s in pieces:
            compact = re.sub(r'[\s\-]+', '', t)
            if re.fullmatch(r'[A-Z]{2,3}', compact): letters.append((compact, s))
            elif re.fullmatch(r'\d{4}', compact): d4.append((compact, s))
            elif re.fullmatch(r'\d{3}', compact): d3.append((compact, s))

        best_l = max(letters, key=lambda x: x[1])[0] if letters else ""
        if d4: best_d, mode = max(d4, key=lambda x: x[1])[0], "4digit"
        elif d3: best_d, mode = max(d3, key=lambda x: x[1])[0], "3digit"
        else: best_d, mode = "", "none"

        if best_l and best_d: return f"{best_l}-{best_d}", f"joined_{mode}"
        return "", "not_found"

    def run_easyocr_on_crop(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0: return "", "empty"
        crop_bgr = self.ensure_3ch(crop_bgr)
        h, w = crop_bgr.shape[:2]
        
        scale = 4
        up = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        x_cut = int(0.22 * w)
        roi = crop_bgr[:, x_cut:] if (w - x_cut) > 20 else crop_bgr
        rh, rw = roi.shape[:2]
        
        line_variants = [roi]
        if rh > 20:
            line_variants.extend([roi[int(0.03*rh):int(0.48*rh), :], roi[int(0.45*rh):int(0.98*rh), :]])

        all_candidates = []
        for img_variant in [up] + line_variants:
            gray = cv2.cvtColor(self.ensure_3ch(img_variant), cv2.COLOR_BGR2GRAY)
            sharp = self._sharpen(self._clahe(gray))
            _, th_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            for stage in [self.ensure_3ch(img_variant), self.ensure_3ch(sharp), self.ensure_3ch(th_otsu)]:
                results = self.easy_reader.readtext(stage, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
                plate, mode = self.extract_plate_from_pieces([r[1] for r in results], [r[2] for r in results])
                if plate and mode != "not_found":
                    all_candidates.append(self.apply_pakistani_corrections(plate))

        if not all_candidates: return "", "not_found"
        return Counter(all_candidates).most_common(1)[0][0], "voted_consensus"

    def process_bay_image(self, bay_img):
        """Takes an image of a parking bay, finds the plate, and reads it."""
        if bay_img is None or bay_img.size == 0: return None
        
        # 1. Detect plate in the bay
        res = self.detector.predict(bay_img, conf=0.25, verbose=False)[0]
        if len(res.boxes) == 0: return None
        
        # 2. Get best box
        boxes = res.boxes.data.cpu().numpy()
        best = max(boxes, key=lambda x: x[4])
        pad = 15
        x1, y1, x2, y2 = max(0, int(best[0])-pad), max(0, int(best[1])-pad), \
                         min(bay_img.shape[1], int(best[2])+pad), min(bay_img.shape[0], int(best[3])+pad)
        
        plate_crop = bay_img[y1:y2, x1:x2]
        
        # 3. Read text
        pred_plate, _ = self.run_easyocr_on_crop(plate_crop)
        return pred_plate if pred_plate else None