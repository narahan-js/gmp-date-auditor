import os
import re
import sys
import json
import base64
import shutil
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import webview

# RapidOCR (오프라인 내장형 고속 OCR 엔진)
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    HAS_OCR = True
except Exception:
    ocr_engine = None
    HAS_OCR = False

MONTH_MAP = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}


def resolve_year(y_str: str) -> int:
    y = int(y_str)
    if len(y_str) == 2:
        return 2000 + y if y <= 79 else 1900 + y
    return y


def normalize_ocr_digits(text: str) -> str:
    text = re.sub(r'(?<![A-Za-z])[OoQ](?=\d)', '0', text)
    text = re.sub(r'(?<=\d)[OoQ](?![A-Za-z])', '0', text)
    return text


def is_valid_date(y: int, m: int, d: int):
    if not (1990 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    try:
        dt = datetime(y, m, d)
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        return None


def extract_dates_from_text(raw_text: str):
    cleaned = normalize_ocr_digits(raw_text)
    results = []
    seen_spans = []

    def add_match(span, matched_str, y, m, d):
        for s, e in seen_spans:
            if not (span[1] <= s or span[0] >= e):
                return
        iso = is_valid_date(y, m, d)
        if iso:
            seen_spans.append(span)
            results.append({'raw': matched_str.strip(), 'iso': iso})

    # 1. YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD
    for m in re.finditer(r'(?<!\d)(20\d{2}|19\d{2})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?(?!\d)', cleaned):
        add_match(m.span(), m.group(0), int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # 2. DD-MMM-YYYY or DD-MMM-YY or DDMMMYY (예: 06-Nov-2026, 06-Nov-26, 06NOV26)
    for m in re.finditer(
        r'(?<![A-Za-z0-9])(\d{1,2})\s*[-./]?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*[-./,]?\s*(\d{2,4})(?![A-Za-z0-9])',
        cleaned,
        re.IGNORECASE
    ):
        mon = MONTH_MAP.get(m.group(2).lower()[:3])
        if mon:
            add_match(m.span(), m.group(0), resolve_year(m.group(3)), mon, int(m.group(1)))

    # 3. MMM DD, YYYY or MMM DD YYYY (예: Nov 06, 2026)
    for m in re.finditer(
        r'(?<![A-Za-z0-9])(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*[-./]?\s*(\d{1,2})\s*[,./-]?\s*(20\d{2}|19\d{2})(?!\d)',
        cleaned,
        re.IGNORECASE
    ):
        mon = MONTH_MAP.get(m.group(1).lower()[:3])
        if mon:
            add_match(m.span(), m.group(0), int(m.group(3)), mon, int(m.group(2)))

    # 4. MM/DD/YYYY or DD/MM/YYYY
    for m in re.finditer(r'(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(20\d{2}|19\d{2})(?!\d)', cleaned):
        p1, p2, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if p1 > 12:
            add_match(m.span(), m.group(0), y, p2, p1)
        else:
            add_match(m.span(), m.group(0), y, p1, p2)

    # 5. YY.MM.DD / YY-MM-DD / YY/MM/DD (예: 26.11.06, 26-11-06)
    for m in re.finditer(r'(?<!\d)(\d{2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)', cleaned):
        add_match(m.span(), m.group(0), resolve_year(m.group(1)), int(m.group(2)), int(m.group(3)))

    return results


def _get_dialog_type(kind: str):
    # pywebview 구버전(OPEN_DIALOG)과 신버전(FileDialog.OPEN) 모두 호환 처리
    if hasattr(webview, 'FileDialog'):
        return webview.FileDialog.OPEN if kind == 'open' else webview.FileDialog.SAVE
    return webview.OPEN_DIALOG if kind == 'open' else webview.SAVE_DIALOG


class AuditorAPI:
    def __init__(self):
        self.work_dir = tempfile.mkdtemp(prefix="gmp_auditor_")
        self.docs = {}

    def select_files(self):
        try:
            window = webview.windows[0]
            file_types = ('PDF Files (*.pdf)', 'All files (*.*)')
            paths = window.create_file_dialog(
                _get_dialog_type('open'),
                allow_multiple=True,
                file_types=file_types
            )
            if not paths:
                return []
            return [self._register_pdf(p) for p in paths]
        except Exception as e:
            return {'error': str(e)}

    def load_dropped_file(self, filename: str, b64_data: str):
        try:
            safe_name = os.path.basename(filename)
            dest = os.path.join(self.work_dir, f"{len(self.docs)+1}_{safe_name}")
            with open(dest, "wb") as f:
                f.write(base64.b64decode(b64_data))
            return self._register_pdf(dest, display_name=safe_name)
        except Exception as e:
            return {'error': str(e)}

    def _register_pdf(self, filepath: str, display_name: str = None):
        doc_id = f"doc_{len(self.docs)+1}_{int(datetime.now().timestamp()*1000)}"
        name = display_name or os.path.basename(filepath)
        pdf = fitz.open(filepath)
        page_count = len(pdf)
        pdf.close()

        self.docs[doc_id] = {
            'id': doc_id,
            'name': name,
            'path': filepath,
            'pageCount': page_count,
            'findings': []
        }
        return {
            'id': doc_id,
            'name': name,
            'pageCount': page_count,
            'hasOcr': HAS_OCR
        }

    def scan_document(self, doc_id: str, cutoff_date: str, use_ocr: bool = True):
        if doc_id not in self.docs:
            return {'error': '문서를 찾을 수 없습니다.'}

        info = self.docs[doc_id]
        pdf = fitz.open(info['path'])
        findings = []
        fid_counter = 1

        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            page_num = page_idx + 1
            pw, ph = page.rect.width, page.rect.height

            page_text = page.get_text("text")
            text_dates = extract_dates_from_text(page_text)
            matched_rects_on_page = []

            for td in text_dates:
                if td['iso'] <= cutoff_date:
                    continue

                rects = page.search_for(td['raw'])
                if not rects:
                    compact = re.sub(r'\s+', '', td['raw'])
                    rects = page.search_for(compact)

                if rects:
                    for r in rects:
                        matched_rects_on_page.append(r)
                        findings.append({
                            'id': f"{doc_id}_f{fid_counter}",
                            'page': page_num,
                            'raw': td['raw'],
                            'iso': td['iso'],
                            'source': 'TEXT',
                            'excluded': False,
                            'normRect': {
                                'x': r.x0 / pw,
                                'y': r.y0 / ph,
                                'w': (r.x1 - r.x0) / pw,
                                'h': (r.y1 - r.y0) / ph
                            }
                        })
                        fid_counter += 1
                else:
                    findings.append({
                        'id': f"{doc_id}_f{fid_counter}",
                        'page': page_num,
                        'raw': td['raw'],
                        'iso': td['iso'],
                        'source': 'TEXT',
                        'excluded': False,
                        'normRect': {'x': 0.05, 'y': 0.05, 'w': 0.25, 'h': 0.03}
                    })
                    fid_counter += 1

            if use_ocr and HAS_OCR:
                has_images = len(page.get_images(full=True)) > 0
                is_scanned_page = len(page_text.strip()) < 40

                if is_scanned_page or has_images:
                    zoom = 220.0 / 72.0
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    img_bytes = pix.tobytes("png")
                    ocr_res, _ = ocr_engine(img_bytes)

                    if ocr_res:
                        img_w, img_h = pix.width, pix.height
                        for box, text, conf in ocr_res:
                            if conf < 0.45:
                                continue
                            ocr_dates = extract_dates_from_text(text)
                            for od in ocr_dates:
                                if od['iso'] <= cutoff_date:
                                    continue

                                xs = [pt[0] for pt in box]
                                ys = [pt[1] for pt in box]
                                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

                                cand_rect = fitz.Rect(
                                    (x0 / img_w) * pw,
                                    (y0 / img_h) * ph,
                                    (x1 / img_w) * pw,
                                    (y1 / img_h) * ph
                                )

                                is_dup = any(
                                    (cand_rect & existing).get_area() > 0.3 * cand_rect.get_area()
                                    for existing in matched_rects_on_page
                                )
                                if is_dup:
                                    continue

                                matched_rects_on_page.append(cand_rect)
                                findings.append({
                                    'id': f"{doc_id}_f{fid_counter}",
                                    'page': page_num,
                                    'raw': od['raw'],
                                    'iso': od['iso'],
                                    'source': 'OCR',
                                    'excluded': False,
                                    'normRect': {
                                        'x': cand_rect.x0 / pw,
                                        'y': cand_rect.y0 / ph,
                                        'w': (cand_rect.x1 - cand_rect.x0) / pw,
                                        'h': (cand_rect.y1 - cand_rect.y0) / ph
                                    }
                                })
                                fid_counter += 1

        pdf.close()
        info['findings'] = findings
        return {'docId': doc_id, 'findings': findings}

    def render_page_image(self, doc_id: str, page_num: int, scale: float = 1.6):
        if doc_id not in self.docs:
            return {'error': '문서 없음'}
        pdf = fitz.open(self.docs[doc_id]['path'])
        page = pdf[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        b64 = base64.b64encode(pix.tobytes("png")).decode('ascii')
        pdf.close()
        return {
            'dataUrl': f"data:image/png;base64,{b64}",
            'width': pix.width,
            'height': pix.height
        }

    def _build_annotated_pdf(self, doc_id: str, findings: list, cutoff_date: str, out_path: str):
        info = self.docs[doc_id]
        pdf = fitz.open(info['path'])
        active = [f for f in findings if not f.get('excluded', False)]

        for f in active:
            p_idx = f['page'] - 1
            if p_idx < 0 or p_idx >= len(pdf):
                continue
            page = pdf[p_idx]
            pw, ph = page.rect.width, page.rect.height
            nr = f['normRect']

            pad_x = 3.0
            pad_y = 2.0
            rect = fitz.Rect(
                max(0, nr['x'] * pw - pad_x),
                max(0, nr['y'] * ph - pad_y),
                min(pw, (nr['x'] + nr['w']) * pw + pad_x),
                min(ph, (nr['y'] + nr['h']) * ph + pad_y)
            )

            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=(0.92, 0.11, 0.14), fill=(1.0, 0.88, 0.25))
            annot.set_border(width=1.6)
            annot.set_opacity(0.42)
            annot.set_info(
                title="GMP Date Auditor",
                content=f"[날짜 역전 검출] 표기: {f['raw']} ({f['iso']}) > 승인기준일: {cutoff_date}"
            )
            annot.update()

        pdf.save(out_path, garbage=3, deflate=True)
        pdf.close()

    def export_single_pdf(self, doc_id: str, findings_json: str, cutoff_date: str):
        if doc_id not in self.docs:
            return {'ok': False, 'msg': '문서를 찾을 수 없습니다.'}
        findings = json.loads(findings_json)
        orig_name = Path(self.docs[doc_id]['name']).stem
        window = webview.windows[0]
        save_path = window.create_file_dialog(
            _get_dialog_type('save'),
            save_filename=f"[DATE_AUDIT]_{orig_name}.pdf",
            file_types=('PDF Files (*.pdf)',)
        )
        if not save_path:
            return {'ok': False, 'msg': '취소됨'}
        if isinstance(save_path, (list, tuple)):
            save_path = save_path[0]

        self._build_annotated_pdf(doc_id, findings, cutoff_date, save_path)
        return {'ok': True, 'path': save_path}

    def export_all_zip(self, all_findings_json: str, cutoff_date: str):
        all_map = json.loads(all_findings_json)
        window = webview.windows[0]
        save_path = window.create_file_dialog(
            _get_dialog_type('save'),
            save_filename=f"GMP_Date_Audit_Report_{cutoff_date}.zip",
            file_types=('ZIP Archive (*.zip)',)
        )
        if not save_path:
            return {'ok': False, 'msg': '취소됨'}
        if isinstance(save_path, (list, tuple)):
            save_path = save_path[0]

        tmp_out = tempfile.mkdtemp(prefix="gmp_zip_")
        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for doc_id, findings in all_map.items():
                if doc_id not in self.docs:
                    continue
                orig_name = Path(self.docs[doc_id]['name']).stem
                pdf_out = os.path.join(tmp_out, f"[DATE_AUDIT]_{orig_name}.pdf")
                self._build_annotated_pdf(doc_id, findings, cutoff_date, pdf_out)
                zf.write(pdf_out, arcname=os.path.basename(pdf_out))

        shutil.rmtree(tmp_out, ignore_errors=True)
        return {'ok': True, 'path': save_path}


def get_resource_path(relative_path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == '__main__':
    api = AuditorAPI()
    html_file = get_resource_path('index.html')
    window = webview.create_window(
        title='GMP Document Date Auditor (오프라인 날짜 역전 검출기)',
        url=html_file,
        js_api=api,
        width=1420,
        height=900,
        min_size=(1100, 720)
    )
    webview.start(debug=False)
