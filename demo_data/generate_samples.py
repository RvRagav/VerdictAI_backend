"""Generate sample PDF documents for VerdictAI demo scenarios.

Creates 4 sample PDFs demonstrating the core evaluation scenarios:
1. sample_nit.pdf — Base NIT with 5 eligibility criteria
2. sample_corrigendum.pdf — Corrigendum amending turnover threshold
3. sample_ca_certificate.pdf — CA certificate with stamp obscuration note
4. sample_bidder_submission.pdf — Bidder submission with parent-company name

Usage:
    python -m backend.demo_data.generate_samples
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER


DEMO_DIR = Path(__file__).parent


def generate_sample_nit():
    """Generate sample NIT PDF with 5 eligibility criteria."""
    pdf_path = DEMO_DIR / "sample_nit.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NITTitle", parent=styles["Title"], alignment=TA_CENTER
    )

    elements = []

    # Header
    elements.append(Paragraph("NOTICE INVITING TENDER", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Tender No: CRPF/SEC-EQUIP/2024-25/001", styles["Heading2"]
    ))
    elements.append(Paragraph(
        "Subject: Supply of Security Equipment for CRPF", styles["Heading3"]
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Department info
    elements.append(Paragraph(
        "Central Reserve Police Force (CRPF)<br/>"
        "Directorate General, CGO Complex, New Delhi<br/>"
        "Category: Security Equipment",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Eligibility criteria section
    elements.append(Paragraph(
        "SECTION IV: ELIGIBILITY CRITERIA", styles["Heading2"]
    ))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "The bidder must satisfy ALL of the following eligibility criteria "
        "to qualify for technical evaluation:",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    # Criterion 1: Numeric threshold
    elements.append(Paragraph(
        "<b>Clause 4.1 — Financial Capacity (Mandatory)</b>", styles["Normal"]
    ))
    elements.append(Paragraph(
        "The bidder shall have an annual turnover of not less than "
        "Rs. 5 Crore (Rupees Five Crore) in each of the last 3 (three) "
        "financial years as per audited balance sheets. "
        "[GFR Rule 173(i) — Override NOT permitted]",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    # Criterion 2: Categorical presence
    elements.append(Paragraph(
        "<b>Clause 4.2 — GST Registration (Mandatory)</b>", styles["Normal"]
    ))
    elements.append(Paragraph(
        "The bidder shall possess a valid GST registration certificate "
        "with active status as on the date of submission. "
        "[GFR Rule 144 — Override NOT permitted]",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    # Criterion 3: Temporal recency
    elements.append(Paragraph(
        "<b>Clause 4.3 — Past Experience (Mandatory)</b>", styles["Normal"]
    ))
    elements.append(Paragraph(
        "The bidder shall have successfully completed a minimum of "
        "3 (three) similar supply orders in the last 5 (five) years "
        "from the date of submission. Completion certificates from "
        "the ordering authority are required as evidence.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    # Criterion 4: Composite
    elements.append(Paragraph(
        "<b>Clause 4.4 — Combined Qualification</b>", styles["Normal"]
    ))
    elements.append(Paragraph(
        "The bidder shall satisfy all of the following sub-criteria: "
        "(a) Annual turnover >= Rs. 10 Crore, "
        "(b) Minimum 5 years operational experience in security equipment domain, "
        "(c) Valid ISO 9001:2015 certification from NABCB-accredited body.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    # Criterion 5: Qualitative assessment
    elements.append(Paragraph(
        "<b>Clause 4.5 — Manufacturing Capacity</b>", styles["Normal"]
    ))
    elements.append(Paragraph(
        "The bidder shall demonstrate adequate manufacturing capacity "
        "for the required quantity as evidenced by plant and machinery list, "
        "production records, and factory inspection reports.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Footer
    elements.append(Paragraph(
        "Date of Issue: 01-Jan-2025<br/>"
        "Last Date for Submission: 15-Feb-2025<br/>"
        "Estimated Cost: Rs. 25 Crore",
        styles["Normal"],
    ))

    doc.build(elements)
    print(f"Generated: {pdf_path}")


def generate_sample_corrigendum():
    """Generate corrigendum PDF amending turnover threshold from 5 Cr to 10 Cr."""
    pdf_path = DEMO_DIR / "sample_corrigendum.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CorTitle", parent=styles["Title"], alignment=TA_CENTER
    )

    elements = []

    # Header
    elements.append(Paragraph("CORRIGENDUM No. 1", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Tender No: CRPF/SEC-EQUIP/2024-25/001", styles["Heading2"]
    ))
    elements.append(Paragraph(
        "Subject: Supply of Security Equipment for CRPF", styles["Heading3"]
    ))
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph(
        "The following amendments are hereby made to the above-mentioned "
        "tender document:",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Amendment table
    table_data = [
        ["Clause", "Original Text", "Amended Text"],
        [
            "4.1",
            "Annual turnover of not less than Rs. 5 Crore",
            "Annual turnover of not less than Rs. 10 Crore (Rupees Ten Crore)",
        ],
    ]

    table = Table(table_data, colWidths=[2 * cm, 7 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph(
        "<b>Reason for Amendment:</b> Based on market assessment and revised "
        "project scope, the financial capacity requirement has been enhanced "
        "to ensure adequate capability of the selected vendor.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph(
        "All other terms and conditions of the original NIT remain unchanged.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph(
        "Date of Corrigendum: 15-Jan-2025<br/>"
        "Revised Last Date for Submission: 28-Feb-2025",
        styles["Normal"],
    ))

    doc.build(elements)
    print(f"Generated: {pdf_path}")


def generate_sample_ca_certificate():
    """Generate CA certificate PDF with simulated stamp obscuration note."""
    pdf_path = DEMO_DIR / "sample_ca_certificate.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CATitle", parent=styles["Title"], alignment=TA_CENTER
    )

    elements = []

    # Header
    elements.append(Paragraph("CHARTERED ACCOUNTANT CERTIFICATE", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Certificate of Annual Turnover", styles["Heading2"]
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # CA details
    elements.append(Paragraph(
        "M/s Sharma & Associates<br/>"
        "Chartered Accountants<br/>"
        "FRN: 012345N<br/>"
        "14, Connaught Place, New Delhi - 110001",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Certificate body
    elements.append(Paragraph(
        "This is to certify that M/s <b>SecureTech Solutions Pvt Ltd</b> "
        "(PAN: AASCS1234F) has achieved the following annual turnover "
        "as per their audited financial statements:",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    # Turnover table
    table_data = [
        ["Financial Year", "Annual Turnover (Rs.)", "Audited By"],
        ["2023-24", "12,45,00,000 (Rs. 12.45 Crore)", "Sharma & Associates"],
        ["2022-23", "10,80,00,000 (Rs. 10.80 Crore)", "Sharma & Associates"],
        ["2021-22", "9,25,00,000 (Rs. 9.25 Crore)", "Sharma & Associates"],
    ]

    table = Table(table_data, colWidths=[4 * cm, 6 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))

    # Stamp obscuration note (simulating what OCR would encounter)
    elements.append(Paragraph(
        "<b>[NOTE: Rubber stamp ink from CA seal overlaps the FY 2021-22 "
        "turnover figure. Red-channel separation required for accurate "
        "extraction. OCR confidence for this region: 0.42]</b>",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph(
        "The above figures are based on audited balance sheets and "
        "profit & loss accounts certified by us.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Signature
    elements.append(Paragraph(
        "Sd/-<br/>"
        "CA Rajesh Sharma<br/>"
        "Partner, M.No. 098765<br/>"
        "Date: 10-Jan-2025<br/>"
        "Place: New Delhi<br/>"
        "UDIN: 25098765ABCDEF1234",
        styles["Normal"],
    ))

    doc.build(elements)
    print(f"Generated: {pdf_path}")


def generate_sample_bidder_submission():
    """Generate bidder submission PDF with parent-company name for entity mismatch demo."""
    pdf_path = DEMO_DIR / "sample_bidder_submission.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "BidTitle", parent=styles["Title"], alignment=TA_CENTER
    )

    elements = []

    # Header — note the parent company name used here
    elements.append(Paragraph("TECHNICAL BID SUBMISSION", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Submitted by: SecureTech Group India Limited", styles["Heading2"]
    ))
    elements.append(Paragraph(
        "(A subsidiary of SecureTech International Holdings)",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Tender No: CRPF/SEC-EQUIP/2024-25/001", styles["Heading3"]
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # Company details — registered name differs from header
    elements.append(Paragraph("COMPANY DETAILS", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * cm))

    details_data = [
        ["Registered Company Name", "SecureTech Solutions Pvt Ltd"],
        ["PAN Number", "AASCS1234F"],
        ["GST Number", "07AASCS1234F1Z5"],
        ["Registered Address", "Plot 45, Sector 62, Noida, UP - 201301"],
        ["Parent Company", "SecureTech Group India Limited"],
        ["Year of Incorporation", "2015"],
    ]

    table = Table(details_data, colWidths=[5 * cm, 10 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))

    # Experience section — uses parent company name
    elements.append(Paragraph("PAST EXPERIENCE", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        "The following supply orders have been completed by "
        "<b>SecureTech Group India Limited</b> (parent company) "
        "in the last 5 years:",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.2 * cm))

    exp_data = [
        ["S.No", "Client", "Order Value", "Completion Date"],
        ["1", "BSF, New Delhi", "Rs. 8.5 Crore", "March 2024"],
        ["2", "CISF, Mumbai", "Rs. 6.2 Crore", "November 2023"],
        ["3", "ITBP, Dehradun", "Rs. 4.8 Crore", "July 2022"],
        ["4", "SSB, Lucknow", "Rs. 3.1 Crore", "January 2022"],
    ]

    exp_table = Table(exp_data, colWidths=[2 * cm, 5 * cm, 4 * cm, 4 * cm])
    exp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(exp_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Note about entity mismatch
    elements.append(Paragraph(
        "<i>Note: All experience certificates are issued in the name of "
        "SecureTech Group India Limited. The bidding entity SecureTech "
        "Solutions Pvt Ltd is a wholly-owned subsidiary.</i>",
        styles["Normal"],
    ))

    doc.build(elements)
    print(f"Generated: {pdf_path}")


def generate_all():
    """Generate all sample PDF documents."""
    print("Generating VerdictAI demo PDFs...")
    print(f"Output directory: {DEMO_DIR}")
    print()

    generate_sample_nit()
    generate_sample_corrigendum()
    generate_sample_ca_certificate()
    generate_sample_bidder_submission()

    print()
    print("All demo PDFs generated successfully.")


if __name__ == "__main__":
    generate_all()
