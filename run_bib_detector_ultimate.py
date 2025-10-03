#!/usr/bin/env python3
"""
run_bib_detector_ultimate.py
Mejor detector de dorsales:
 - usa YOLO (detecta dígitos)
 - agrupa dígitos en dorsales candidatos
 - extrae ROI, preprocesa y aplica Tesseract (solo dígitos)
 - confirma dorsal por estabilidad temporal + confianza OCR
 - guarda en resultados.xlsx (Posición, Dorsal, Timestamp)
"""

import cv2
import numpy as np
import argparse
import os
import sys
import pandas as pd
import pytesseract
import imutils
from datetime import datetime
from collections import deque
from math import ceil

# ------------------------ util imagen ------------------------
def clamp(v, a, b): return max(a, min(b, v))

def expand_bbox(bbox, frame_w, frame_h, pad=0.15):
    x, y, w, h = bbox
    padx = int(w * pad)
    pady = int(h * pad)
    x1 = clamp(x - padx, 0, frame_w-1)
    y1 = clamp(y - pady, 0, frame_h-1)
    x2 = clamp(x + w + padx, 0, frame_w-1)
    y2 = clamp(y + h + pady, 0, frame_h-1)
    return (x1, y1, x2-x1, y2-y1)

def deskew_and_preprocess(roi):
    # roi: BGR image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Resize to improve OCR accuracy
    h, w = gray.shape
    scale = 1.0
    if w < 300:
        scale = 300 / float(w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # Bilateral denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Threshold (adaptive)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    # Morphology to join components horizontally (digits)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Deskew using minAreaRect of non-zero pixels
    coords = cv2.findNonZero(th)
    if coords is not None and len(coords) > 50:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        th = cv2.warpAffine(th, M, (th.shape[1], th.shape[0]), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    # Final resize to reasonable width for tesseract
    final_w = 600
    if gray.shape[1] < final_w:
        r = final_w / gray.shape[1]
        gray = cv2.resize(gray, (final_w, int(gray.shape[0]*r)), interpolation=cv2.INTER_LINEAR)
        th = cv2.resize(th, (final_w, int(th.shape[0]*r)), interpolation=cv2.INTER_LINEAR)
    # Return both versions: processed gray and binary inverted (white text on black)
    return gray, th

def ocr_digits(image_gray, image_bin, tesseract_config):
    """
    Run pytesseract on the preprocessed images and return:
     - digits_str : string of digits (only 0-9)
     - avg_conf : average confidence (0-100), or -1 if none
    """
    # prefer image_bin for OCR, but also try gray as fallback
    try:
        data = pytesseract.image_to_data(image_bin, config=tesseract_config, output_type=pytesseract.Output.DICT)
    except Exception:
        data = pytesseract.image_to_data(image_gray, config=tesseract_config, output_type=pytesseract.Output.DICT)
    texts = []
    confs = []
    n = len(data['text'])
    for i in range(n):
        txt = str(data['text'][i]).strip()
        conf_raw = data.get('conf', [])[i] if 'conf' in data else -1
        try:
            conf = int(conf_raw)
        except Exception:
            conf = -1
        if txt != "" and any(ch.isdigit() for ch in txt):
            # keep only digits
            digits = ''.join([c for c in txt if c.isdigit()])
            if digits != "":
                texts.append((data['left'][i] if 'left' in data else 0, digits))
                if conf >= 0:
                    confs.append(conf)
    # order by x (left)
    texts_sorted = [d for x,d in sorted(texts, key=lambda t:t[0])]
    digits_str = ''.join(texts_sorted)
    avg_conf = (sum(confs)/len(confs)) if confs else -1
    return digits_str, avg_conf

# ------------------------ YOLO helpers ------------------------
def load_yolo(cfg_path, weights_path, names_path, use_cuda=False):
    if not (os.path.exists(cfg_path) and os.path.exists(weights_path) and os.path.exists(names_path)):
        raise FileNotFoundError("cfg/weights/names faltan o ruta incorrecta")
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    if use_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names_path, 'r', encoding='utf-8', errors='ignore') as f:
        classes = [l.strip() for l in f.readlines() if l.strip()]
    ln = net.getLayerNames()
    try:
        out_layers = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    except:
        out_layers = [ln[i-1] for i in net.getUnconnectedOutLayers()]
    return net, out_layers, classes

def postprocess_raw(frame, outputs, conf_threshold=0.55, nms_threshold=0.4):
    H, W = frame.shape[:2]
    boxes, confs, classIDs = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            if len(scores)==0: continue
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf >= conf_threshold:
                cx, cy, w, h = det[0:4]
                x = int((cx - w/2)*W); y = int((cy - h/2)*H)
                bw = int(w*W); bh = int(h*H)
                x = clamp(x, 0, W-1); y = clamp(y, 0, H-1)
                bw = clamp(bw, 0, W-x); bh = clamp(bh, 0, H-y)
                boxes.append([x,y,bw,bh]); confs.append(conf); classIDs.append(cid)
    if not boxes: return []
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, nms_threshold)
    results = []
    if len(idxs)>0:
        flat = idxs.flatten() if hasattr(idxs,'flatten') else [i for sub in idxs for i in sub]
        for i in flat:
            results.append({'bbox': boxes[i], 'conf': confs[i], 'classID': classIDs[i]})
    return results

# ------------------------ Trackers ------------------------
class SimpleTracker:
    def __init__(self, bbox, label, history_len=6):
        self.bbox = bbox[:]  # [x,y,w,h]
        self.label_history = deque([label], maxlen=history_len)
        self.ocr_history = deque([], maxlen=history_len)
        self.ocr_conf_history = deque([], maxlen=history_len)
        self.missed = 0
        self.saved = False
    def update_bbox(self, bbox):
        # union to expand
        x1 = min(self.bbox[0], bbox[0])
        y1 = min(self.bbox[1], bbox[1])
        x2 = max(self.bbox[0]+self.bbox[2], bbox[0]+bbox[2])
        y2 = max(self.bbox[1]+self.bbox[3], bbox[1]+bbox[3])
        self.bbox = [x1, y1, x2-x1, y2-y1]
    def add_label(self, lbl):
        self.label_history.append(lbl)
        self.missed = 0
    def add_ocr(self, txt, conf):
        if txt is not None:
            self.ocr_history.append(txt)
        if conf is not None:
            self.ocr_conf_history.append(conf)
    def mark_missed(self):
        self.missed += 1
    def most_common_label(self):
        if not self.label_history: return ""
        cnt = {}
        for v in self.label_history:
            cnt[v] = cnt.get(v,0)+1
        return max(cnt.items(), key=lambda x:x[1])[0]
    def most_common_ocr(self):
        if not self.ocr_history: return ""
        cnt = {}
        for v in self.ocr_history:
            cnt[v] = cnt.get(v,0)+1
        return max(cnt.items(), key=lambda x:x[1])[0]
    def avg_ocr_conf(self):
        if not self.ocr_conf_history: return -1
        return sum(self.ocr_conf_history)/len(self.ocr_conf_history)

# ------------------------ Main ------------------------
def check_cv2_gui():
    """Return True if cv2.imshow works in this build, False otherwise."""
    try:
        test = np.zeros((10,10,3), dtype=np.uint8)
        cv2.namedWindow('cv2_test', cv2.WINDOW_NORMAL)
        cv2.imshow('cv2_test', test)
        cv2.waitKey(1)
        cv2.destroyWindow('cv2_test')
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.55)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--frames_confirm", type=int, default=4)
    p.add_argument("--history_len", type=int, default=6)
    p.add_argument("--excel", default="resultados.xlsx")
    p.add_argument("--tess-path", default="", help="(Windows) ruta a tesseract.exe si no está en PATH")
    p.add_argument("--use-cuda", action="store_true")
    args = p.parse_args()

    # tesseract path
    if args.tess_path:
        pytesseract.pytesseract.tesseract_cmd = args.tess_path

    net, out_layers, classes = load_yolo(args.cfg, args.weights, args.names, use_cuda=args.use_cuda)
    print("[INFO] Modelo cargado.")

    # create excel if not exists
    if not os.path.exists(args.excel):
        pd.DataFrame(columns=["Posición","Dorsal","Timestamp"]).to_excel(args.excel,index=False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara.")
        sys.exit(1)

    trackers = []
    saved = set()
    frame_i = 0

    # tesseract config: digits only, single line
    tess_config = "--psm 7 -c tessedit_char_whitelist=0123456789"

    use_gui = check_cv2_gui()
    if not use_gui:
        print("[WARN] OpenCV no tiene soporte GUI en este build. Las ventanas no funcionarán; se guardarán imágenes de visualización en 'output/' .")
    os.makedirs('output', exist_ok=True)

    print("[INFO] Iniciando. Presiona 'q' para salir.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_i += 1
            H, W = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(out_layers)
            raw = postprocess_raw(frame, outputs, conf_threshold=args.conf, nms_threshold=args.nms)
            # convert raw dets -> list of {(bbox,label,conf)}
            dets = []
            for r in raw:
                cid = r['classID']
                label = classes[cid] if cid < len(classes) else str(cid)
                dets.append({'bbox': r['bbox'], 'label': label, 'conf': r['conf']})

            # If no detections: mark trackers missed
            if not dets:
                for tr in trackers:
                    tr.mark_missed()
                trackers = [t for t in trackers if t.missed <= args.frames_confirm*3]
            else:
                # grouping digits into candidate dorsal groups (heuristic)
                dets_sorted = sorted(dets, key=lambda d: d['bbox'][0])
                groups = []
                for d in dets_sorted:
                    assigned = False
                    for g in groups:
                        # proximity heuristic
                        bx = d['bbox'][0] + d['bbox'][2]/2
                        gx = g['bbox'][0] + g['bbox'][2]/2
                        horiz = abs(bx - gx)
                        maxw = max(d['bbox'][2], g['bbox'][2])
                        if horiz < 1.8*maxw:
                            # vertical center similarity
                            by = d['bbox'][1] + d['bbox'][3]/2
                            gy = g['bbox'][1] + g['bbox'][3]/2
                            if abs(by-gy) < max(0.6*max(d['bbox'][3], g['bbox'][3]), 20):
                                # assign
                                g['members'].append(d)
                                g['bbox'] = (min(g['bbox'][0], d['bbox'][0]),
                                             min(g['bbox'][1], d['bbox'][1]),
                                             max(g['bbox'][0]+g['bbox'][2], d['bbox'][0]+d['bbox'][2]) - min(g['bbox'][0], d['bbox'][0]),
                                             max(g['bbox'][1]+g['bbox'][3], d['bbox'][1]+d['bbox'][3]) - min(g['bbox'][1], d['bbox'][1]))
                                assigned = True
                                break
                    if not assigned:
                        groups.append({'bbox': d['bbox'][:], 'members':[d]})

                # For each group, compute concatenated label (by x), crop ROI, OCR
                group_results = []
                for g in groups:
                    members_sorted = sorted(g['members'], key=lambda m: m['bbox'][0])
                    concat_label = ''.join([m['label'] for m in members_sorted])
                    gx, gy, gw, gh = map(int, g['bbox'])
                    ex_gx, ex_gy, ex_gw, ex_gh = expand_bbox((gx,gy,gw,gh), W, H, pad=0.2)
                    roi = frame[ex_gy:ex_gy+ex_gh, ex_gx:ex_gx+ex_gw]
                    if roi.size == 0:
                        ocr_str, ocr_conf = "", -1
                    else:
                        gray, th = deskew_and_preprocess(roi)
                        ocr_str, ocr_conf = ocr_digits(gray, th, tess_config)
                        # fallback: if OCR returns nothing, try on gray
                        if (ocr_str=="" or (ocr_conf!=-1 and ocr_conf<10)) and gray is not None:
                            ocr_str2, ocr_conf2 = ocr_digits(gray, gray, tess_config)
                            if ocr_str2 and (ocr_conf2>ocr_conf):
                                ocr_str, ocr_conf = ocr_str2, ocr_conf2
                    group_results.append({'bbox': (gx,gy,gw,gh), 'concat': concat_label, 'ocr': ocr_str, 'ocr_conf': ocr_conf})

                # Match groups to trackers by IoU-like overlap
                matched = set()
                for gr in group_results:
                    matched_idx = None
                    for i, tr in enumerate(trackers):
                        # compute IoU-like using intersection area
                        if iou_bbox(gr['bbox'], tr.bbox) >= 0.20:
                            matched_idx = i; break
                    if matched_idx is not None:
                        tr = trackers[matched_idx]
                        tr.update_bbox(gr['bbox'])
                        tr.add_label(gr['concat'])
                        tr.add_ocr(gr['ocr'], gr['ocr_conf'])
                        matched.add(matched_idx)
                    else:
                        newt = SimpleTracker(list(gr['bbox']), gr['concat'], history_len=args.history_len)
                        newt.add_ocr(gr['ocr'], gr['ocr_conf'])
                        trackers.append(newt)
                # trackers not matched: mark missed
                for idx, tr in enumerate(trackers):
                    if idx not in matched:
                        tr.mark_missed()
                # prune old trackers
                trackers = [t for t in trackers if t.missed <= args.frames_confirm*3]

            # check for trackers that are stable and not saved
            for tr in trackers:
                if tr.saved: continue
                # consider OCR if available
                most_ocr = tr.most_common_ocr()
                most_label = tr.most_common_label()
                avg_conf = tr.avg_ocr_conf()
                # decide acceptance:
                accept = False
                final_num = ""
                # prefer OCR if stable and confidence good
                if most_ocr and avg_conf is not None and avg_conf >= 45 and tr.ocr_history and len(tr.ocr_history) >= args.frames_confirm:
                    final_num = ''.join([c for c in most_ocr if c.isdigit()])
                    if final_num:
                        accept = True
                # else fallback to concatenated labels if stable across label history
                if (not accept) and most_label and len(tr.label_history) >= args.frames_confirm:
                    # ensure the label is consistent in history
                    if tr.label_history.count(most_label) >= args.frames_confirm:
                        final_num = ''.join([c for c in most_label if c.isdigit()])
                        if final_num:
                            accept = True
                # additional filter: if final_num short (one digit) and we want avoid singles, require stronger confidence
                if accept and len(final_num) == 1:
                    # require higher OCR confidence or more stability
                    if avg_conf < 65 and tr.label_history.count(most_label) < (args.frames_confirm+1):
                        accept = False

                if accept:
                    if final_num not in saved:
                        # append to excel
                        df = pd.read_excel(args.excel)
                        pos = len(df) + 1
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        df.loc[len(df)] = [pos, final_num, ts]
                        df.to_excel(args.excel, index=False)
                        saved.add(final_num)
                        tr.saved = True
                        print(f"[SAVED] Pos {pos} | Dorsal {final_num} | conf={avg_conf:.1f} | history_labels={list(tr.label_history)} | history_ocr={list(tr.ocr_history)}")
                    else:
                        tr.saved = True

            # Visualization for debug
            vis = frame.copy()
            for tr in trackers:
                x,y,w,h = map(int,tr.bbox)
                color = (0,200,0) if not tr.saved else (255,128,0)
                cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
                txt = tr.most_common_ocr() or tr.most_common_label()
                cv2.putText(vis, f"{txt}", (x, max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis, f"Guardados: {len(saved)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            if use_gui:
                cv2.imshow("Ultimate Dorsales", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # save periodic visualization (every 30 frames)
                if frame_i % 30 == 0:
                    out_file = os.path.join('output', f'vis_{frame_i:06d}.jpg')
                    cv2.imwrite(out_file, vis)
                    print(f"[OUT] Guardada visualizacion: {out_file}")

    finally:
        cap.release()
        if use_gui:
            cv2.destroyAllWindows()
        print("[INFO] Finalizado. Guardados:", len(saved))

# small helper IoU for bbox lists
def iou_bbox(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    w = max(0, x2-x1); h = max(0, y2-y1)
    inter = w*h
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/union if union>0 else 0.0

if __name__ == "__main__":
    main()
