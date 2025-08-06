
# InternVL3-2B Key-Value Extraction Evaluation Report
## Executive Summary

**Generated:** 2025-08-06 21:24:43
**Evaluation Dataset:** 20 synthetic business documents
**Model:** InternVL3-2B Vision-Language Model
**Task:** Structured key-value extraction from business documents

### 🎯 Overall Performance Summary

**Average Accuracy:** 36.9%
- Perfect Extractions (≥99%): 0/20 (0.0%)
- Good Extractions (80-98%): 0/20 (0.0%)
- Fair Extractions (60-79%): 0/20 (0.0%)
- Poor Extractions (<60%): 20/20 (100.0%)

### 📊 Key Findings

1. **Model Reliability:** InternVL3-2B demonstrates consistent performance across diverse document types
2. **Field Extraction:** Successfully extracts 0 out of 25 fields with ≥90% accuracy
3. **Best Performance:** synthetic_invoice_004.png (52.0% accuracy)
4. **Challenging Cases:** synthetic_invoice_013.png (20.0% accuracy)

### 🏆 Top Performing Fields
 1. DOCUMENT_TYPE        77.0%
 2. INVOICE_DATE         75.0%
 3. SUBTOTAL             75.0%
 4. SUPPLIER             75.0%
 5. TOTAL                75.0%
 6. GST                  70.0%
 7. ABN                  55.0%
 8. BUSINESS_ADDRESS     50.0%
 9. PAYER_NAME           50.0%
10. PAYER_EMAIL          45.0%

### 📈 Deployment Readiness Assessment

**Overall Grade:** C (Needs Improvement)

**Recommendations:**
- ❌ **NOT READY FOR PRODUCTION:** Significant accuracy improvements needed
- 🔧 **REQUIRES INVESTIGATION:** Review model configuration and training data
- 📋 **HUMAN VALIDATION REQUIRED:** Manual review necessary for all extractions

### 📋 Technical Details
- **Extraction Fields:** 25 structured business document fields
- **Document Types:** Invoices, receipts, bank statements, tax documents
- **Evaluation Method:** Sophisticated field-specific comparison with tolerance for numeric fields
- **Data Quality:** Ground truth validated against 20 diverse synthetic business documents

---
*Report generated automatically by InternVL3 evaluation system*
