
import os
import sys
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple, List
import time

import cv2
import numpy as np

# -------------------------
# Tesseract discovery
# -------------------------
def find_tesseract_executable() -> Optional[str]:
    env_path = os.environ.get("TESSERACT_EXE")
    if env_path and os.path.isfile(env_path):
        return env_path
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.isfile(win_default):
        return win_default
    return None

# -------------------------
# Image preprocessing (strong)
# -------------------------
class ImagePreprocessor:
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def gamma_correct(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        invGamma = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        if len(image.shape) == 3:
            return cv2.LUT(image, table)
        else:
            return cv2.LUT(image, table)

    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    @staticmethod
    def upscale_and_pad(image: np.ndarray, scale: float = 2.0, pad: int = 20) -> np.ndarray:
        up = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if len(up.shape) == 2:
            return cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        else:
            return cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))

    @staticmethod
    def morphological_operations(bin_img: np.ndarray) -> np.ndarray:
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
        return connected

    @staticmethod
    def detect_and_correct_skew(gray_or_bin: np.ndarray) -> Tuple[np.ndarray, float]:
        img = gray_or_bin if len(gray_or_bin.shape) == 2 else cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=100, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                if -45 < angle < 45:
                    angles.append(angle)
            if len(angles)>0:
                median_angle = float(np.median(angles))
                if abs(median_angle) > 0.5:
                    h,w = img.shape[:2]
                    center = (w//2, h//2)
                    rot = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(gray_or_bin, rot, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated, median_angle
        return gray_or_bin, 0.0

    @staticmethod
    def multi_threshold(gray: np.ndarray) -> List[np.ndarray]:
        results = []
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(otsu)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, g_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(g_otsu)
        ag = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
        results.append(ag)
        am = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 12)
        results.append(am)
        _, inv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        results.append(inv_otsu)
        return results

    @classmethod
    def preprocess_for_ocr(cls, image_bgr: np.ndarray, method: str = "ensemble") -> Tuple[List[np.ndarray], float]:
        img = cls.enhance_contrast(image_bgr.copy())
        img = cls.gamma_correct(img, gamma=1.2)
        img = cls.denoise_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        gray = cls.upscale_and_pad(gray, scale=2.0, pad=20)
        corrected, skew = cls.detect_and_correct_skew(gray)
        if method == "fast":
            bilateral = cv2.bilateralFilter(corrected, 9, 75, 75)
            _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned = cls.morphological_operations(thresh)
            return [cleaned], skew
        ths = cls.multi_threshold(corrected)
        results = [cls.morphological_operations(t) for t in ths]
        return results, skew

# -------------------------
# Document boundary detection (auto)
# -------------------------
def detect_document_boundary(image_bgr: np.ndarray) -> np.ndarray:
    """Find largest 4-point contour and warp it. If not found, return original."""
    orig = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            pts = approx.reshape(4,2)
            rect = np.zeros((4,2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            (tl,tr,br,bl) = rect
            width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
            height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
            dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, M, (width, height))
            return warped
    return orig

# -------------------------
# Tesseract runner and EasyOCR fallback
# -------------------------
def _run_tesseract_with_confidence(image: np.ndarray, tesseract_exe: str, lang: str, psm: int) -> Tuple[str, float]:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        if not cv2.imwrite(tmp_path, image):
            raise RuntimeError("Failed to write temporary image.")
        cmd = [tesseract_exe, tmp_path, "stdout", "--psm", str(psm), "--oem", "3", "-l", lang, "tsv", "-c", "preserve_interword_spaces=1"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Tesseract error: {proc.stderr.strip()}")
        lines = proc.stdout.strip().split("\n")
        if len(lines) < 2:
            return "", 0.0
        parts = []
        confs = []
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) >= 12:
                conf = cols[10]
                txt = cols[11]
                if conf != "-1" and txt.strip():
                    try:
                        cval = float(conf)
                        if cval > 0:
                            parts.append(txt)
                            confs.append(cval)
                    except ValueError:
                        pass
        if not parts:
            return "", 0.0
        return " ".join(parts).strip(), float(np.mean(confs)) if confs else 0.0
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def try_easyocr_fallback(image_bgr: np.ndarray, langs: List[str]) -> Tuple[str, float]:
    try:
        import easyocr  # type: ignore
    except Exception:
        return "", 0.0
    try:
        reader = easyocr.Reader(langs, gpu=False)
        results = reader.readtext(image_bgr)
        parts = []
        confs = []
        for bbox, text, conf in results:
            if text.strip():
                parts.append(text.strip())
                confs.append(conf * 100.0)
        if parts:
            return " ".join(parts), float(np.mean(confs) if confs else 0.0)
        return "", 0.0
    except Exception:
        return "", 0.0

def ocr_image_strong(image_bgr: np.ndarray, tesseract_exe: str, lang: str = "eng", use_ensemble: bool = True, use_easyocr_fallback: bool = True) -> Tuple[str, float, dict]:
    pre = ImagePreprocessor()
    mode = "ensemble" if use_ensemble else "fast"
    processed_images, skew = pre.preprocess_for_ocr(image_bgr, mode)
    debug = {"skew_angle": skew, "methods_tried": len(processed_images), "results": []}
    psm_modes = [6,11,7,13,4,3] if use_ensemble else [6]
    best_text = ""
    best_score = -1.0
    best_conf = 0.0
    for i, img in enumerate(processed_images):
        for psm in psm_modes:
            try:
                text, conf = _run_tesseract_with_confidence(img, tesseract_exe, lang, psm)
                length = len(text.strip())
                score = conf * (1.0 + min(length/100.0, 0.5))
                debug["results"].append({"method": i, "psm": psm, "confidence": conf, "length": length, "preview": text[:120].replace("\n"," ")})
                if length>0 and score>best_score:
                    best_score = score; best_text = text; best_conf = conf
            except Exception as e:
                debug["results"].append({"method": i, "psm": psm, "error": str(e)})
    if (not best_text.strip()) and use_easyocr_fallback:
        fb_text, fb_conf = try_easyocr_fallback(image_bgr, langs=[lang.split('+')[0]])
        debug["easyocr_fallback"] = bool(fb_text.strip())
        if fb_text.strip():
            best_text, best_conf = fb_text, fb_conf
    return best_text, best_conf, debug

# -------------------------
# ROI selection via mouse
# -------------------------
roi_start = None
roi_end = None
selecting_roi = False
roi_active = None  # (x1,y1,x2,y2) or None

def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, selecting_roi, roi_active
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        selecting_roi = True
        roi_end = None
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        selecting_roi = False
        if roi_start and roi_end:
            x1,y1 = roi_start; x2,y2 = roi_end
            roi_active = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))

# -------------------------
# Main app loop
# -------------------------
def main():
    # Initialize ROI-related state before any usage
    roi_active = False
    roi_points = []
    roi_rect = None
    roi_selected = False

    tesseract_exe = find_tesseract_executable()
    if not tesseract_exe:
        print("Error: Tesseract not found. Set TESSERACT_EXE or install tesseract.", file=sys.stderr)
        sys.exit(1)
    print(f"Using Tesseract at: {tesseract_exe}")
    print("Controls: q=quit, o=OCR, f=fast OCR, d=save debug images, g=gray, t=thresh, c=color, r=real-time, b=toggle boundary, m=draw/clear ROI, s=save frame")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name=='nt' else 0)
    if not cap.isOpened():
        print("Error: Could not open camera.", file=sys.stderr); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Enhanced OCR Camera (STRONG)")
    cv2.setMouseCallback("Enhanced OCR Camera (STRONG)", mouse_callback)

    show_mode = "color"
    last_frame = None
    real_time = False
    last_ocr_time = 0.0
    use_boundary = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame."); continue
            last_frame = frame.copy()

            display = frame.copy()
            # show ROI selection rectangle while dragging
            if selecting_roi and roi_start and roi_end:
                cv2.rectangle(display, roi_start, roi_end, (0,255,0), 2)

            # show active roi outline
            if roi_active:
                x1,y1,x2,y2 = roi_active
                cv2.rectangle(display, (x1,y1), (x2,y2), (255,0,0), 2)

            if show_mode == "gray":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif show_mode == "thresh":
                pre = ImagePreprocessor()
                proc_imgs, _ = pre.preprocess_for_ocr(frame, "fast")
                if proc_imgs:
                    disp = cv2.cvtColor(proc_imgs[0], cv2.COLOR_GRAY2BGR)
                else:
                    disp = frame
            else:
                disp = display

            # Prepare frame_for_ocr: apply boundary detection and ROI cropping
            frame_for_ocr = frame.copy()
            if use_boundary:
                try:
                    doc = detect_document_boundary(frame_for_ocr)
                    if doc is not None:
                        frame_for_ocr = doc
                except Exception as e:
                    print("Boundary detection error:", e)
            if roi_active:
                try:
                    x1,y1,x2,y2 = roi_active
                    # clamp
                    h,w = frame_for_ocr.shape[:2]
                    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
                    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
                    if x2>x1 and y2>y1:
                        frame_for_ocr = frame_for_ocr[y1:y2, x1:x2]
                except Exception as e:
                    print("ROI crop error:", e)

            # Real-time OCR throttle
            if real_time and (time.time() - last_ocr_time > 2.0):
                try:
                    text, conf, _ = ocr_image_strong(frame_for_ocr, tesseract_exe, lang="eng", use_ensemble=False)
                    current_text = text
                    current_confidence = conf
                    last_ocr_time = time.time()
                except Exception as e:
                    print("Real-time OCR error:", e)
            # If current_text exists, overlay it
            if 'current_text' in locals() and current_text:
                overlay = disp.copy()
                cv2.rectangle(overlay, (10, disp.shape[0]-120), (disp.shape[1]-10, disp.shape[0]-10),(0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)
                font = cv2.FONT_HERSHEY_SIMPLEX
                lines = current_text.splitlines()
                y = disp.shape[0]-95
                shown = 0
                for line in lines:
                    if shown>=3: break
                    txt = (line[:68]+"...") if len(line)>71 else line
                    cv2.putText(disp, txt, (18,y), font, 0.6, (0,255,0),1, cv2.LINE_AA)
                    y += 28; shown +=1
                conf_text = f"Confidence: {current_confidence:.1f}%"
                cv2.putText(disp, conf_text, (18, disp.shape[0]-20), font, 0.6, (255,255,255),1, cv2.LINE_AA)

            cv2.imshow("Enhanced OCR Camera (STRONG)", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): break
            elif key == ord('g'): show_mode='gray'; print("gray preview")
            elif key == ord('t'): show_mode='thresh'; print("thresh preview")
            elif key == ord('c'): show_mode='color'; print("color preview")
            elif key == ord('r'): real_time = not real_time; print("real-time", real_time)
            elif key == ord('s'):
                if last_frame is not None:
                    cv2.imwrite("frame.png", last_frame); print("Saved frame.png")
            elif key == ord('d'):
                if last_frame is not None:
                    print("Saving debug images...")
                    pre = ImagePreprocessor()
                    imgs, skew = pre.preprocess_for_ocr(last_frame, "ensemble")
                    for i, im in enumerate(imgs):
                        cv2.imwrite(f"debug_processed_{i}.png", im)
                    print(f"Saved {len(imgs)} debug images (skew {skew:.2f}°)")
            elif key == ord('f'):
                if last_frame is None: print("No frame"); continue
                try:
                    print("Running FAST OCR...")
                    start = time.time()
                    text, conf, dbg = ocr_image_strong(last_frame, tesseract_exe, lang="eng", use_ensemble=False)
                    print("FAST OCR:", f"Time {time.time()-start:.2f}s", f"Conf {conf:.1f}%"); print(text if text else "(no text)")
                except Exception as e:
                    print("FAST OCR failed:", e)
            elif key == ord('o'):
                if last_frame is None: print("No frame"); continue
                try:
                    print("Running STRONG OCR...")
                    start = time.time()
                    text, conf, dbg = ocr_image_strong(last_frame, tesseract_exe, lang="eng", use_ensemble=True)
                    print("STRONG OCR:", f"Time {time.time()-start:.2f}s", f"Conf {conf:.1f}%"); print(text if text else "(no text)")
                except Exception as e:
                    print("STRONG OCR failed:", e)
            elif key == ord('b'):
                use_boundary = not use_boundary; print("Auto boundary detection", "on" if use_boundary else "off")
            elif key == ord('m'):
                # toggle clear/manual mode: if roi_active exists, clear it; otherwise instruct user to draw
                if roi_active:
                    print("Clearing manual ROI"); roi_active = None
                else:
                    print("Draw ROI with mouse (click-drag). Press 'm' again to clear.")
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
