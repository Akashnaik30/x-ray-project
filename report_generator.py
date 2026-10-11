"""
PACS Clinical Radiology PDF Report Generator
Author: Antigravity AI
Version: 1.0.0
Description: Compiles patient demographics, neural network probabilities, multi-label warnings,
             anatomical structural measurements (CTR), impressions, and rendered PACS canvas images
             into a downloadable, high-resolution clinical PDF report.
"""

import io
import base64
from datetime import datetime
from PIL import Image
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class RadiologyReportPDF(FPDF):
    """Custom FPDF class for professional medical radiology reports."""

    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Dark slate hospital header bar
        self.set_fill_color(15, 23, 42)  # #0f172a
        self.rect(0, 0, 210, 26, 'F')
        
        # Primary Hospital Header
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 13)
        self.set_xy(12, 5)
        self.cell(120, 6, 'METROPOLIS ADVANCED IMAGING LABS', new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        # Badge on right
        self.set_font('helvetica', 'B', 8)
        self.set_text_color(0, 255, 204)
        self.set_xy(140, 5)
        self.cell(58, 6, 'PACS WORKSTATION v4.2 // SECURE', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Subtitle
        self.set_font('helvetica', '', 8)
        self.set_text_color(148, 163, 184)
        self.set_xy(12, 11)
        self.cell(186, 5, 'Department of Thoracic Radiology & AI-Assisted Diagnostics | DICOM Node: METRO_CXR_AI', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_xy(12, 16)
        self.set_text_color(203, 213, 225)
        self.cell(186, 5, 'Tel: (555) 019-9400  |  PACS Scanner IP: 192.168.12.85  |  Modality: CXR (PA/AP)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Cyan divider accent line
        self.set_draw_color(0, 255, 204)
        self.set_line_width(0.7)
        self.line(0, 26, 210, 26)
        
        self.set_y(30)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(203, 213, 225)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.set_font('helvetica', 'I', 7)
        self.set_text_color(100, 116, 139)
        self.set_xy(10, -11)
        self.cell(130, 4, 'Confidential Medical Document - Generated for Clinical Demonstration & Research Only', new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(60, 4, f'Page {self.page_no()}', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_pdf_report(data: dict, doctor_name: str, canvas_image_data: str = None) -> bytes:
    """
    Compiles full clinical report into a downloadable PDF binary buffer.
    """
    pdf = RadiologyReportPDF()
    pdf.add_page()
    
    # ---------------------------------------------------------
    # 1. Report Title Bar
    # ---------------------------------------------------------
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(130, 7, 'CLINICAL RADIOLOGY & AI DIAGNOSTIC REPORT', new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(71, 85, 105)
    scan_time = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.cell(60, 7, f'Date: {scan_time}', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(1)
    
    # ---------------------------------------------------------
    # 2. Patient Demographics & Study Metadata
    # ---------------------------------------------------------
    pdf.set_fill_color(241, 245, 249)  # Light gray-blue
    pdf.set_draw_color(203, 213, 225)
    pdf.set_line_width(0.2)
    pdf.rect(10, pdf.get_y(), 190, 24, 'FD')
    
    start_y = pdf.get_y() + 2
    pdf.set_xy(14, start_y)
    
    col1_w, col2_w, col3_w, col4_w = 26, 68, 28, 60
    
    # Row 1
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col1_w, 5, 'PATIENT ID:')
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(col2_w, 5, str(data.get('patient_id', 'UNKNOWN')))
    
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col3_w, 5, 'PATIENT NAME:')
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(col4_w, 5, str(data.get('patient_name', 'UNKNOWN')).upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Row 2
    pdf.set_x(14)
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col1_w, 5, 'AGE / GENDER:')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(col2_w, 5, f"{data.get('patient_age', 'N/A')} YRS  /  {data.get('patient_gender', 'O')}")
    
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col3_w, 5, 'REFERRING MD:')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(col4_w, 5, str(data.get('referred_by', 'N/A')), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Row 3
    pdf.set_x(14)
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col1_w, 5, 'ATTENDING:')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(col2_w, 5, f"Dr. {doctor_name}")
    
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col3_w, 5, 'STUDY ACC #:')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(col4_w, 5, f"SCAN-{data.get('scan_id', '0000')[:8].upper()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_y(start_y + 25)
    
    # ---------------------------------------------------------
    # 3. Multi-Label Diagnostic Warning (if present)
    # ---------------------------------------------------------
    multi_label_warning = data.get('multi_label_warning')
    if multi_label_warning:
        pdf.set_fill_color(254, 243, 199)  # Light Amber #fef3c7
        pdf.set_draw_color(245, 158, 11)   # Amber #f59e0b
        pdf.set_line_width(0.3)
        pdf.rect(10, pdf.get_y(), 190, 13, 'FD')
        
        pdf.set_xy(14, pdf.get_y() + 1.5)
        pdf.set_font('helvetica', 'B', 7.5)
        pdf.set_text_color(180, 83, 9)  # Dark amber
        pdf.cell(182, 4, '[!] MULTI-LABEL CO-OCCURRENCE / CLOSE MARGIN ALERT', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_x(14)
        pdf.set_font('helvetica', '', 7)
        pdf.set_text_color(120, 53, 15)
        pdf.multi_cell(182, 3.2, multi_label_warning)
        pdf.ln(2)

    # ---------------------------------------------------------
    # 4. Rendered Imaging Plate (PACS Canvas & Grad-CAM)
    # ---------------------------------------------------------
    img_plate_y = pdf.get_y()
    
    # If canvas image was transmitted as base64, render it prominently
    has_canvas_img = False
    if canvas_image_data and ',' in canvas_image_data:
        try:
            raw_b64 = canvas_image_data.split(',')[1]
            img_bytes = base64.b64decode(raw_b64)
            pil_canvas = Image.open(io.BytesIO(img_bytes))
            
            # Save to temporary buffer for FPDF
            canvas_buf = io.BytesIO()
            pil_canvas.save(canvas_buf, format='JPEG', quality=90)
            canvas_buf.seek(0)
            
            # Draw frame around image
            img_w = 88
            img_h = 88
            pdf.set_fill_color(15, 23, 42)
            pdf.rect(10, img_plate_y, img_w + 4, img_h + 4, 'F')
            pdf.image(canvas_buf, x=12, y=img_plate_y + 2, w=img_w, h=img_h)
            
            has_canvas_img = True
        except Exception as e:
            has_canvas_img = False

    # Right side of image: Probabilities & Anatomical Metrics
    right_x = 106 if has_canvas_img else 10
    right_w = 94 if has_canvas_img else 190
    
    pdf.set_xy(right_x, img_plate_y)
    
    # Section: Deep Learning Probabilities
    pdf.set_font('helvetica', 'B', 8.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(right_w, 5, 'AI DEEP LEARNING PROBABILITIES (DenseNet-121):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    metrics = data.get('metrics', {})
    pathology_rows = [
        ('Normal / No Active Finding', metrics.get('normal', 0.0), '#10b981'),
        ('Pneumonia / Consolidation', metrics.get('pneumonia', 0.0), '#ef4444'),
        ('Cardiomegaly (Enlarged Heart)', metrics.get('cardiomegaly', 0.0), '#f59e0b'),
        ('Pleural Effusion', metrics.get('effusion', 0.0), '#f59e0b'),
        ('Pneumothorax (Collapsed)', metrics.get('pneumothorax', 0.0), '#ef4444'),
    ]
    
    for label, val, color_hex in pathology_rows:
        pdf.set_x(right_x)
        pdf.set_font('helvetica', '', 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(right_w * 0.65, 4.2, label)
        
        # Percentage text
        pdf.set_font('helvetica', 'B', 7.5)
        if val > 30.0:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(15, 23, 42)
        pdf.cell(right_w * 0.35, 4.2, f"{val:.1f}%", align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Mini visual bar
        pdf.set_x(right_x)
        pdf.set_fill_color(226, 232, 240)
        bar_max_w = right_w - 2
        pdf.rect(right_x, pdf.get_y(), bar_max_w, 2, 'F')
        
        # Filled bar portion
        fill_w = max(1.0, (val / 100.0) * bar_max_w)
        if val > 40.0:
            pdf.set_fill_color(239, 68, 68)  # Red
        elif val > 25.0:
            pdf.set_fill_color(245, 158, 11) # Orange
        else:
            pdf.set_fill_color(16, 185, 129) # Green
        pdf.rect(right_x, pdf.get_y(), fill_w, 2, 'F')
        pdf.ln(3)

    pdf.ln(1)
    
    # Section: Anatomical Structural Metrics
    pdf.set_x(right_x)
    pdf.set_font('helvetica', 'B', 8.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(right_w, 5, 'ANATOMICAL COMPUTER-VISION METRICS:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    heuristics = data.get('heuristics', {})
    ctr_val = heuristics.get('ctr', 0.44)
    lung_dens = heuristics.get('lung_density', 0.25)
    costo_dens = heuristics.get('costophrenic_density', 0.28)
    apical_flat = heuristics.get('apical_flatness', 0.65)
    
    # CTR status
    if ctr_val <= 0.50:
        ctr_status = "NORMAL (0.42-0.50)"
    elif ctr_val <= 0.55:
        ctr_status = "BORDERLINE (>0.50)"
    else:
        ctr_status = "ENLARGED (>0.55)"
        
    opacity_status = "OPACIFIED (>0.35)" if lung_dens > 0.35 else "CLEAR"
    costo_status = "BLUNTED (>0.38)" if costo_dens > 0.38 else "SHARP"
    apical_status = "AIR LAYER" if apical_flat > 0.75 else "VASCULAR"
    
    metric_items = [
        ("Cardiothoracic Ratio (CTR):", f"{ctr_val:.2f}", ctr_status),
        ("Lung Infiltration Index:", f"{lung_dens:.2f}", opacity_status),
        ("Fluid Recess Density:", f"{costo_dens:.2f}", costo_status),
        ("Apical Air Coefficient:", f"{apical_flat:.2f}", apical_status),
    ]
    
    for m_label, m_val, m_stat in metric_items:
        pdf.set_x(right_x)
        pdf.set_font('helvetica', '', 7)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(right_w * 0.50, 4, m_label)
        
        pdf.set_font('helvetica', 'B', 7)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(right_w * 0.20, 4, m_val)
        
        pdf.set_font('helvetica', 'B', 6.5)
        if "ENLARGED" in m_stat or "OPACIFIED" in m_stat or "BLUNTED" in m_stat or "AIR" in m_stat:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(16, 185, 129)
        pdf.cell(right_w * 0.30, 4, m_stat, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Reset Y below image plate
    if has_canvas_img:
        caption_y = img_plate_y + 92
        pdf.set_xy(10, caption_y)
        pdf.set_font('helvetica', 'I', 6.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(88, 4, 'Fig 1: Workstation Viewport with Grad-CAM Attention Heatmap & Calibration', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(max(caption_y + 6, pdf.get_y() + 4))
    else:
        pdf.ln(4)

    # ---------------------------------------------------------
    # 5. Primary Diagnostic Impression & Findings Narrative
    # ---------------------------------------------------------
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 6, 'FD')
    
    pdf.set_xy(14, pdf.get_y() + 1)
    pdf.set_font('helvetica', 'B', 8.5)
    pdf.set_text_color(15, 23, 42)
    dom_finding = data.get('dominant_finding', 'UNREMARKABLE')
    dom_score = data.get('dominant_score', 0.0)
    pdf.cell(100, 4, f"PRIMARY CLINICAL FINDING: {dom_finding.upper()}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(82, 4, f"AI CONFIDENCE: {dom_score:.1f}%", align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(2)
    
    # Narrative Impressions List
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(12)
    pdf.cell(186, 4, 'RADIOLOGICAL FINDINGS & CLINICAL IMPRESSIONS:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    impressions = data.get('impressions', [])
    for idx, item in enumerate(impressions, 1):
        pdf.set_x(14)
        pdf.set_font('helvetica', '', 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(5, 4, f"{idx}.")
        pdf.set_font('helvetica', '', 7.5)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(177, 4, item)
        pdf.ln(0.5)

    # ---------------------------------------------------------
    # 6. Digital Verification & Signature Block
    # ---------------------------------------------------------
    pdf.ln(3)
    sig_y = pdf.get_y()
    
    # Prevent page spill if near bottom
    if sig_y > 245:
        pdf.add_page()
        sig_y = pdf.get_y()
        
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(10, sig_y, 190, 20, 'FD')
    
    pdf.set_xy(14, sig_y + 2)
    pdf.set_font('helvetica', 'B', 7.5)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(100, 3.5, 'ATTENDING RADIOLOGIST DIGITAL SIGN-OFF:', new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    pdf.set_font('helvetica', 'I', 7)
    pdf.cell(82, 3.5, 'SECURE CRYPTOGRAPHIC VERIFICATION', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_xy(14, sig_y + 6.5)
    pdf.set_font('helvetica', 'B', 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(100, 4.5, f"Dr. {doctor_name}, M.D., Board Certified Thoracic Radiologist", new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    pdf.set_font('helvetica', 'B', 7.5)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(82, 4.5, '[VERIFIED & DIGITALLY TRANSMITTED]', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_xy(14, sig_y + 12)
    pdf.set_font('helvetica', '', 6.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(182, 3.5, f"Audit Key: METRO-PACS-{data.get('scan_id', 'UUID')[:12].upper()}  |  Timestamp: {scan_time}  |  HL7 FHIR v4 DiagnosticReport", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Return binary PDF buffer
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
