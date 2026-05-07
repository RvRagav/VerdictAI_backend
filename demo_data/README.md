# VerdictAI Demo Data

This directory contains sample PDF documents for demonstrating VerdictAI's four core scenarios.

## Demo Scenarios

### 1. Base NIT with Corrigendum (`sample_nit.pdf` + `sample_corrigendum.pdf`)

- **NIT**: Contains 5 eligibility criteria for "Supply of Security Equipment for CRPF"
  - Annual turnover ≥ Rs. 5 Crore (numeric_threshold)
  - Valid GST registration (categorical_presence)
  - 3 similar supply orders in last 5 years (temporal_recency)
  - Turnover + experience + ISO certification (composite)
  - Adequate manufacturing capacity (qualitative_assessment)
- **Corrigendum**: Amends the turnover threshold from Rs. 5 Crore to Rs. 10 Crore

This demonstrates ETS version assembly and corrigendum diff display.

### 2. CA Certificate with Stamp Obscuration (`sample_ca_certificate.pdf`)

- A Chartered Accountant certificate showing annual turnover figures
- Contains a simulated note about rubber stamp ink overlapping text
- Demonstrates the stamp separation pipeline and degraded OCR confidence routing

### 3. Bidder Submission with Entity Mismatch (`sample_bidder_submission.pdf`)

- Submission documents from "SecureTech Solutions Pvt Ltd"
- Documents reference parent company "SecureTech Group India Limited"
- Demonstrates entity matcher detecting parent-company name mismatch
- Routes to mandatory review regardless of numeric confidence

### 4. CPM Precedent Injection

- Uses the seeded CPM corpus with ambiguous criterion language
- Demonstrates how past officer decisions inform current evaluations
- No separate PDF needed — uses the NIT criteria with CPM bootstrap data

## Generating Sample PDFs

Run the generation script:

```bash
cd backend
python -m demo_data.generate_samples
```

This creates all sample PDFs in this directory using reportlab.

## Notes

- These are text-based PDFs (not scanned images) for prototype purposes
- The OCR pipeline treats them as already-extracted text
- Real deployment would use actual scanned documents with Tesseract OCR
