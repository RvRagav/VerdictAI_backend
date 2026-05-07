"""Generate production-realistic Indian government tender PDFs for demo.

Produces:
  - sample_nit_crpf.pdf         Full CRPF NIT with 8 eligibility criteria, multi-page, structured clauses
  - sample_corrigendum.pdf      Corrigendum amending turnover threshold
  - sample_bidder_good.pdf      Compliant bidder submission with all supporting evidence
  - sample_bidder_mismatch.pdf  Bidder with parent-company entity mismatch (fraud signal)
  - sample_bidder_weak.pdf      Bidder with insufficient turnover (should FAIL)
  - sample_ca_certificate.pdf   CA certificate with stamp obscuration simulation

Run: python -m backend.demo_data.generate_realistic_samples
"""

from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DEMO_DIR = Path(__file__).parent


def _styles() -> dict:
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=ss["Title"], alignment=TA_CENTER,
            fontSize=14, spaceAfter=12, textColor=colors.black,
        ),
        "h1": ParagraphStyle(
            "H1", parent=ss["Heading1"],
            fontSize=13, spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#0f172a"),
        ),
        "h2": ParagraphStyle(
            "H2", parent=ss["Heading2"],
            fontSize=11, spaceBefore=10, spaceAfter=6,
        ),
        "clause": ParagraphStyle(
            "Clause", parent=ss["Normal"],
            fontSize=10, alignment=TA_JUSTIFY, spaceAfter=6,
            leading=14,
        ),
        "body": ParagraphStyle(
            "Body", parent=ss["Normal"],
            fontSize=10, alignment=TA_JUSTIFY, spaceAfter=5,
            leading=13,
        ),
        "stamp": ParagraphStyle(
            "Stamp", parent=ss["Normal"],
            fontSize=9, textColor=colors.red, alignment=TA_CENTER,
            spaceBefore=4, spaceAfter=4,
        ),
    }


# ─── 1. Full NIT with 8 criteria ─────────────────────────────────────────


def generate_sample_nit_crpf() -> Path:
    path = DEMO_DIR / "sample_nit_crpf.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    s = _styles()
    elements: list = []

    # Header
    elements.append(Paragraph("NOTICE INVITING TENDER", s["title"]))
    elements.append(Paragraph(
        "Central Reserve Police Force (CRPF)<br/>"
        "Directorate General | CGO Complex, New Delhi - 110003",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    nit_meta = [
        ["Tender No:", "CRPF/DG/PROV-VI/SEC-EQUIP/2024-25/001"],
        ["Date of Issue:", "15 January 2025"],
        ["Mode of Bidding:", "Two-Bid System (Technical + Financial)"],
        ["Estimated Value:", "Rs. 25.00 Crore"],
        ["EMD:", "Rs. 50,00,000 (Rupees Fifty Lakhs)"],
        ["Last Date for Submission:", "15 February 2025, 15:00 hrs IST"],
        ["Date of Technical Bid Opening:", "18 February 2025, 11:00 hrs IST"],
    ]
    t = Table(nit_meta, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph(
        "Subject: Procurement of Advanced Perimeter Security Equipment "
        "including Thermal Imaging Cameras, Motion Sensors, and Integrated "
        "Surveillance Systems for CRPF Installations across multiple zones.",
        s["body"],
    ))

    elements.append(PageBreak())

    # SECTION IV — Eligibility criteria
    elements.append(Paragraph("SECTION IV — ELIGIBILITY CRITERIA", s["h1"]))
    elements.append(Paragraph(
        "The bidder must satisfy all of the following pre-qualification criteria "
        "as per GFR 2017 Rule 173(i). Submissions not meeting the mandatory "
        "criteria shall be summarily rejected without further evaluation.",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    # Criterion 1: Numeric — turnover
    elements.append(Paragraph("Clause 4.1 — Financial Capacity (Mandatory)", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall have an annual turnover of not less than "
        "Rs. 10 (Ten) Crore in each of the last 3 (three) financial years, "
        "as per audited balance sheets certified by a practising Chartered "
        "Accountant. The turnover shall pertain to the bidding entity's own "
        "operations; consolidated figures from parent or subsidiary companies "
        "are not permissible. "
        "[GFR Rule 173(i) — Override NOT permitted]",
        s["clause"],
    ))

    # Criterion 2: Numeric — net worth
    elements.append(Paragraph("Clause 4.2 — Net Worth (Mandatory)", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall have a positive net worth of not less than "
        "Rs. 2 (Two) Crore as on the last date of the immediately preceding "
        "financial year, calculated from the audited balance sheet. "
        "[GFR Rule 173(i)]",
        s["clause"],
    ))

    # Criterion 3: Categorical — GST
    elements.append(Paragraph("Clause 4.3 — GST Registration (Mandatory)", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall possess a valid GST registration certificate with "
        "active status as on the date of bid submission. The GSTIN shall be "
        "in the name of the bidding entity. Certificates in the name of any "
        "parent, subsidiary, or affiliate entity shall not be accepted. "
        "[GFR Rule 144]",
        s["clause"],
    ))

    # Criterion 4: Categorical — PAN
    elements.append(Paragraph("Clause 4.4 — Permanent Account Number (Mandatory)", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall furnish a valid Permanent Account Number (PAN) "
        "issued by the Income Tax Department in the name of the bidding "
        "entity. [GFR Rule 144]",
        s["clause"],
    ))

    # Criterion 5: Temporal — similar works
    elements.append(Paragraph("Clause 4.5 — Past Experience (Mandatory)", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall have successfully completed a minimum of 3 (three) "
        "similar supply orders in the last 5 (five) financial years from the "
        "date of bid submission. For the purposes of this clause, 'similar' "
        "shall mean supply orders involving security and surveillance "
        "equipment to any Central Armed Police Force, paramilitary "
        "organisation, State Police, or defence establishment. "
        "Each qualifying supply order shall be of value not less than "
        "Rs. 3 (Three) Crore individually. Completion certificates issued "
        "by the ordering authority are required as evidence.",
        s["clause"],
    ))

    elements.append(PageBreak())

    # Criterion 6: Categorical — ISO
    elements.append(Paragraph("Clause 4.6 — Quality Management Certification", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall possess a valid ISO 9001:2015 certification for "
        "Quality Management Systems, issued by a certification body "
        "accredited by NABCB (National Accreditation Board for "
        "Certification Bodies) or equivalent. The scope of certification "
        "shall cover the manufacture and/or supply of the products being "
        "tendered.",
        s["clause"],
    ))

    # Criterion 7: Categorical — MSME (preferential)
    elements.append(Paragraph("Clause 4.7 — MSME Registration (Preferential)", s["h2"]))
    elements.append(Paragraph(
        "Bidders registered under MSME / Udyam Registration are eligible for "
        "preferential treatment under the Public Procurement Policy for "
        "Micro and Small Enterprises 2012, subject to meeting all mandatory "
        "criteria. A copy of the Udyam Registration Certificate shall be "
        "submitted if preferential treatment is claimed.",
        s["clause"],
    ))

    # Criterion 8: Qualitative — manufacturing capacity
    elements.append(Paragraph("Clause 4.8 — Manufacturing and Delivery Capability", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall demonstrate adequate manufacturing capacity and "
        "supply-chain capability to deliver the tendered quantity within the "
        "stipulated timeline of 180 days from the date of Purchase Order. "
        "Evidence shall include plant and machinery list, existing production "
        "capacity certificates, details of source of components, and a "
        "signed delivery schedule commitment.",
        s["clause"],
    ))

    # Debarment declaration
    elements.append(Paragraph("Clause 4.9 — Debarment Declaration (Mandatory)", s["h2"]))
    elements.append(Paragraph(
        "The bidder shall submit a self-declaration that the bidding entity, "
        "its directors, or any of its promoters are not currently debarred, "
        "blacklisted, or under any suspension order issued by the Central "
        "Vigilance Commission, Government of India, any Central or State "
        "government, GeM portal, or any defence or paramilitary "
        "establishment. [GFR Rule 151]",
        s["clause"],
    ))

    elements.append(Spacer(1, 0.4*cm))
    elements.append(Paragraph(
        "<i>Note: All values in this document are as amended from time to "
        "time. Refer corrigendum, if any, issued by the Procurement "
        "Authority for revised thresholds.</i>",
        s["body"],
    ))

    doc.build(elements)
    print(f"Generated: {path}")
    return path


# ─── 2. Corrigendum ──────────────────────────────────────────────────────


def generate_sample_corrigendum() -> Path:
    path = DEMO_DIR / "sample_corrigendum.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm)
    s = _styles()
    elements: list = []

    elements.append(Paragraph("CORRIGENDUM No. 1", s["title"]))
    elements.append(Paragraph(
        "Central Reserve Police Force (CRPF)<br/>"
        "Directorate General, New Delhi",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    meta = [
        ["Tender No:", "CRPF/DG/PROV-VI/SEC-EQUIP/2024-25/001"],
        ["Corrigendum Date:", "25 January 2025"],
        ["Revised Last Date:", "28 February 2025, 15:00 hrs"],
    ]
    t = Table(meta, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph(
        "Subject: Supply of Advanced Perimeter Security Equipment — "
        "Corrigendum to original NIT dated 15 January 2025.",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph(
        "The following amendments are hereby made to the above-mentioned "
        "NIT. All other terms and conditions of the original tender shall "
        "remain unchanged.",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    # Amendment table
    rows = [
        ["Clause", "Original Text", "Amended Text"],
        [
            "4.1",
            "Annual turnover of not less than Rs. 10 (Ten) Crore "
            "in each of the last 3 financial years",
            "Annual turnover of not less than Rs. 15 (Fifteen) Crore "
            "in each of the last 3 financial years",
        ],
        [
            "4.5",
            "Each qualifying supply order shall be of value not less "
            "than Rs. 3 (Three) Crore individually",
            "Each qualifying supply order shall be of value not less "
            "than Rs. 5 (Five) Crore individually",
        ],
    ]
    t = Table(rows, colWidths=[2*cm, 7*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph(
        "<b>Reason for Amendment:</b> Based on market assessment and revised "
        "project scope, the financial capacity and experience thresholds "
        "have been enhanced to ensure adequate capability of the selected "
        "vendor.",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph(
        "Signed:<br/>"
        "Inspector General of Police (Provisioning)<br/>"
        "For and on behalf of the Directorate General, CRPF",
        s["body"],
    ))

    doc.build(elements)
    print(f"Generated: {path}")
    return path


# ─── 3. Compliant bidder (all criteria pass) ─────────────────────────────


def generate_bidder_good() -> Path:
    path = DEMO_DIR / "sample_bidder_good.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm)
    s = _styles()
    elements: list = []

    elements.append(Paragraph("TECHNICAL BID SUBMISSION", s["title"]))
    elements.append(Paragraph("Sentinel Defence Systems Pvt Ltd", s["h1"]))
    elements.append(Paragraph(
        "Plot 45, Electronic City Phase II, Bengaluru - 560100<br/>"
        "Tender No: CRPF/DG/PROV-VI/SEC-EQUIP/2024-25/001",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    # Company details
    elements.append(Paragraph("Company Details", s["h2"]))
    details = [
        ["Registered Name", "Sentinel Defence Systems Pvt Ltd"],
        ["CIN", "U32109KA2015PTC087654"],
        ["PAN", "AACCS9876K"],
        ["GSTIN", "29AACCS9876K1Z3"],
        ["Udyam Registration", "UDYAM-KA-05-0087654"],
        ["ISO Certification", "ISO 9001:2015 (Cert No. QMS/2024/1122)"],
        ["Year of Incorporation", "2015"],
        ["Authorised Signatory", "Ravi Narayanan, Managing Director"],
    ]
    t = Table(details, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    # Financials (pass criteria 4.1 and 4.2)
    elements.append(Paragraph("Financial Summary (Audited)", s["h2"]))
    fin_rows = [
        ["Financial Year", "Annual Turnover (Rs.)", "Net Worth (Rs.)"],
        ["2023-24", "Rs. 18.45 Crore", "Rs. 4.20 Crore"],
        ["2022-23", "Rs. 16.80 Crore", "Rs. 3.75 Crore"],
        ["2021-22", "Rs. 15.25 Crore", "Rs. 3.10 Crore"],
    ]
    t = Table(fin_rows, colWidths=[4*cm, 6*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    # Past projects (pass criterion 4.5)
    elements.append(Paragraph("Past Supply Orders (Last 5 Years)", s["h2"]))
    proj_rows = [
        ["Description", "Order Value", "Completion Date"],
        ["Thermal imaging cameras for BSF, Pathankot sector", "Rs. 8.40 Crore", "12-Mar-2024"],
        ["Motion sensors for CISF industrial security", "Rs. 6.25 Crore", "25-Nov-2023"],
        ["Integrated surveillance for ITBP, Dehradun", "Rs. 7.80 Crore", "08-Jul-2022"],
        ["Night vision devices for SSB, Lucknow", "Rs. 5.50 Crore", "18-Feb-2022"],
    ]
    t = Table(proj_rows, colWidths=[8*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    # Manufacturing capacity (pass criterion 4.8)
    elements.append(Paragraph("Manufacturing Capability", s["h2"]))
    elements.append(Paragraph(
        "Sentinel Defence operates a 45,000 sq ft manufacturing facility in "
        "Electronic City, Bengaluru, with a monthly production capacity of "
        "500 integrated surveillance units. The facility is equipped with "
        "SMT assembly lines, automated testing stations, and a dedicated "
        "thermal imaging calibration lab. Current workforce includes 85 "
        "engineers and 120 production technicians. The plant is certified "
        "for defence-grade equipment manufacturing. We confirm our ability "
        "to deliver the tendered quantity within the stipulated 180-day "
        "timeline from the date of Purchase Order.",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    # Debarment declaration (pass criterion 4.9)
    elements.append(Paragraph("Declaration", s["h2"]))
    elements.append(Paragraph(
        "<b>We, Sentinel Defence Systems Pvt Ltd, hereby declare that our "
        "company, its directors, and promoters are NOT currently debarred, "
        "blacklisted, or under any suspension order issued by CVC, GeM, or "
        "any Central/State government body.</b>",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        "Authorised Signature: Ravi Narayanan, Managing Director<br/>"
        "Date: 12 February 2025 | Place: Bengaluru",
        s["body"],
    ))

    doc.build(elements)
    print(f"Generated: {path}")
    return path


# ─── 4. Entity mismatch bidder (parent company substitution fraud) ───────


def generate_bidder_mismatch() -> Path:
    path = DEMO_DIR / "sample_bidder_mismatch.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm)
    s = _styles()
    elements: list = []

    elements.append(Paragraph("TECHNICAL BID SUBMISSION", s["title"]))
    elements.append(Paragraph(
        "ApexGuard Technologies Pvt Ltd",
        s["h1"],
    ))
    elements.append(Paragraph(
        "<i>(A wholly-owned subsidiary of ApexGuard Group International Ltd)</i><br/>"
        "Tender No: CRPF/DG/PROV-VI/SEC-EQUIP/2024-25/001",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    # The ambiguous entity — notice the bidding entity is
    # "ApexGuard Technologies Pvt Ltd" but several supporting documents
    # reference the parent "ApexGuard Group International Ltd"
    elements.append(Paragraph("Company Details", s["h2"]))
    details = [
        ["Bidding Entity (Registered)", "ApexGuard Technologies Pvt Ltd"],
        ["Bidder PAN", "AAACA1234F"],
        ["Bidder GSTIN", "07AAACA1234F1Z8"],
        ["Year of Incorporation", "2021"],
        ["Parent Company", "ApexGuard Group International Ltd"],
        ["Parent Company PAN", "AAACA0000X"],
    ]
    t = Table(details, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    # Financials — in the PARENT company's name (this is the fraud signal)
    elements.append(Paragraph(
        "Financial Summary (Audited) — ApexGuard Group International Ltd",
        s["h2"],
    ))
    elements.append(Paragraph(
        "<i>Note: The following figures pertain to the consolidated accounts "
        "of <b>ApexGuard Group International Ltd</b> (parent), not to "
        "ApexGuard Technologies Pvt Ltd (bidder). The subsidiary was "
        "incorporated in 2021 and does not yet have 3 years of independent "
        "turnover history.</i>",
        s["body"],
    ))
    elements.append(Spacer(1, 0.15*cm))

    fin_rows = [
        ["Financial Year", "Consolidated Turnover (Rs.)", "Net Worth (Rs.)"],
        ["2023-24", "Rs. 24.50 Crore", "Rs. 6.80 Crore"],
        ["2022-23", "Rs. 22.10 Crore", "Rs. 5.90 Crore"],
        ["2021-22", "Rs. 19.75 Crore", "Rs. 4.85 Crore"],
    ]
    t = Table(fin_rows, colWidths=[4*cm, 6*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    # Past projects in parent's name
    elements.append(Paragraph(
        "Past Supply Orders (executed by ApexGuard Group International Ltd)",
        s["h2"],
    ))
    proj_rows = [
        ["Description", "Order Value", "Completion Date"],
        ["Perimeter security for ITBP, Leh (ApexGuard Group)",
         "Rs. 6.80 Crore", "14-Jan-2024"],
        ["Surveillance systems for BSF, Jodhpur (ApexGuard Group)",
         "Rs. 5.40 Crore", "22-Sep-2023"],
        ["Security equipment for CISF, Mumbai (ApexGuard Group)",
         "Rs. 4.90 Crore", "11-Apr-2022"],
    ]
    t = Table(proj_rows, colWidths=[8*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph(
        "We hereby confirm that ApexGuard Technologies Pvt Ltd shares the "
        "manufacturing facilities, technical teams, and quality systems of "
        "our parent ApexGuard Group International Ltd, and will execute "
        "this order using those shared resources.",
        s["body"],
    ))

    doc.build(elements)
    print(f"Generated: {path}")
    return path


# ─── 5. Weak bidder (turnover too low) ───────────────────────────────────


def generate_bidder_weak() -> Path:
    path = DEMO_DIR / "sample_bidder_weak.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm)
    s = _styles()
    elements: list = []

    elements.append(Paragraph("TECHNICAL BID SUBMISSION", s["title"]))
    elements.append(Paragraph("Nexus Security Solutions Pvt Ltd", s["h1"]))
    elements.append(Paragraph(
        "Industrial Area Phase I, Chandigarh - 160002<br/>"
        "Tender No: CRPF/DG/PROV-VI/SEC-EQUIP/2024-25/001",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph("Company Details", s["h2"]))
    details = [
        ["Registered Name", "Nexus Security Solutions Pvt Ltd"],
        ["PAN", "AACCN5432L"],
        ["GSTIN", "04AACCN5432L1Z9"],
        ["Year of Incorporation", "2018"],
    ]
    t = Table(details, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    # Turnover below threshold (fails 4.1)
    elements.append(Paragraph("Financial Summary", s["h2"]))
    fin_rows = [
        ["Financial Year", "Annual Turnover (Rs.)", "Net Worth (Rs.)"],
        ["2023-24", "Rs. 8.20 Crore", "Rs. 1.45 Crore"],
        ["2022-23", "Rs. 6.50 Crore", "Rs. 1.10 Crore"],
        ["2021-22", "Rs. 5.80 Crore", "Rs. 0.90 Crore"],
    ]
    t = Table(fin_rows, colWidths=[4*cm, 6*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph("Past Supply Orders", s["h2"]))
    proj_rows = [
        ["Description", "Order Value", "Completion Date"],
        ["Small-ticket CCTV for local municipality",
         "Rs. 1.20 Crore", "15-May-2023"],
        ["Residential complex security installation",
         "Rs. 0.80 Crore", "22-Jan-2023"],
    ]
    t = Table(proj_rows, colWidths=[8*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)

    doc.build(elements)
    print(f"Generated: {path}")
    return path


# ─── 6. CA certificate with stamp obscuration ────────────────────────────


def generate_ca_certificate_stamp() -> Path:
    path = DEMO_DIR / "sample_ca_certificate_stamp.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm)
    s = _styles()
    elements: list = []

    elements.append(Paragraph("CHARTERED ACCOUNTANT CERTIFICATE", s["title"]))
    elements.append(Paragraph("Certificate of Annual Turnover", s["h2"]))
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph(
        "M/s Sharma &amp; Associates<br/>"
        "Chartered Accountants | FRN: 012345N<br/>"
        "14, Connaught Place, New Delhi - 110001",
        s["body"],
    ))
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph(
        "This is to certify that <b>Sentinel Defence Systems Pvt Ltd</b> "
        "(PAN: AACCS9876K) has achieved the following annual turnover as "
        "per their audited financial statements:",
        s["body"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    fin_rows = [
        ["Financial Year", "Annual Turnover (Rs.)", "Audited By"],
        ["2023-24", "Rs. 18,45,00,000 (Rs. 18.45 Cr)", "Sharma &amp; Associates"],
        ["2022-23", "Rs. 16,80,00,000 (Rs. 16.80 Cr)", "Sharma &amp; Associates"],
        ["2021-22", "Rs. 15,25,00,000 (Rs. 15.25 Cr)", "Sharma &amp; Associates"],
    ]
    t = Table(fin_rows, colWidths=[4*cm, 7*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4*cm))

    # Simulate stamp obscuration in the way the L1 pipeline's red-channel
    # stamp detector would actually see it — a red rectangle overlapping
    # part of the text.
    elements.append(Paragraph(
        "<font color='red'>⬛ [RUBBER STAMP: OFFICE SEAL "
        "— SHARMA &amp; ASSOC. — NEW DELHI — AUTHENTIC ⬛ "
        "Overlaps FY 2021-22 turnover figure]</font>",
        s["stamp"],
    ))
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph(
        "The above figures are based on audited balance sheets and profit "
        "&amp; loss accounts certified by us. UDIN: 25098765ABCDEF1234.",
        s["body"],
    ))
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph(
        "<b>CA Rajesh Sharma</b><br/>"
        "Partner, M.No. 098765<br/>"
        "Date: 10 January 2025 | Place: New Delhi",
        s["body"],
    ))

    doc.build(elements)
    print(f"Generated: {path}")
    return path


# ─── Main ────────────────────────────────────────────────────────────────


def generate_all() -> None:
    print(f"Generating realistic VerdictAI demo PDFs...")
    print(f"Output directory: {DEMO_DIR}\n")
    generate_sample_nit_crpf()
    generate_sample_corrigendum()
    generate_bidder_good()
    generate_bidder_mismatch()
    generate_bidder_weak()
    generate_ca_certificate_stamp()
    print("\nAll demo PDFs generated successfully.")


if __name__ == "__main__":
    generate_all()
