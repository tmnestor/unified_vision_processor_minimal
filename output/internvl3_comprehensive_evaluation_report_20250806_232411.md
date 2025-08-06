
# InternVL3-2B Key-Value Extraction Evaluation Report
## Executive Summary

**Generated:** 2025-08-06 23:24:11
**Evaluation Dataset:** 20 synthetic business documents
**Model:** InternVL3-2B Vision-Language Model
**Task:** Structured key-value extraction from business documents

### 🎯 Overall Performance Summary

**Average Accuracy:** 74.9%
- Perfect Extractions (≥99%): 0/20 (0.0%)
- Good Extractions (80-98%): 7/20 (35.0%)
- Fair Extractions (60-79%): 12/20 (60.0%)
- Poor Extractions (<60%): 1/20 (5.0%)

### 📊 Key Findings

1. **Model Reliability:** InternVL3-2B demonstrates consistent performance across diverse document types
2. **Field Extraction:** Successfully extracts 10 out of 25 fields with ≥90% accuracy
3. **Best Performance:** synthetic_invoice_020.png (87.2% accuracy)
4. **Challenging Cases:** synthetic_invoice_002.png (59.2% accuracy)

### 🏆 Top Performing Fields
 1. ACCOUNT_HOLDER       100.0%
 2. BANK_NAME            100.0%
 3. BSB_NUMBER           100.0%
 4. PAYER_EMAIL          100.0%
 5. STATEMENT_PERIOD     100.0%
 6. SUPPLIER             100.0%
 7. GST                  95.0%
 8. SUBTOTAL             95.0%
 9. TOTAL                95.0%
10. DUE_DATE             90.0%

### 📈 Deployment Readiness Assessment

**Overall Grade:** B (Fair)

**Recommendations:**
- ⚠️ **REQUIRES OPTIMIZATION:** Consider fine-tuning or prompt engineering
- ⚠️ **INCREASED OVERSIGHT:** Human validation recommended for important documents
- 🔧 **PILOT DEPLOYMENT:** Suitable for pilot programs with close monitoring

### 📋 Technical Details
- **Extraction Fields:** 25 structured business document fields
- **Document Types:** Invoices, receipts, bank statements, tax documents
- **Evaluation Method:** Sophisticated field-specific comparison with tolerance for numeric fields
- **Data Quality:** Ground truth validated against 20 diverse synthetic business documents

---
*Report generated automatically by InternVL3 evaluation system*
