# Invoice Key-Value Extraction Prompt

    Extract all key-value pairs from this invoice document. Return the information in the following format:

    DOCUMENT_TYPE: [value or N/A]
    SUPPLIER_NAME: [Supplier name]
    SUPPLIER_ADDRESS: [Full address]
    SUPPLIER_ABN: [11-digit Australian Business Number or N/A]
    SUPPLIER_PHONE: [Phone number]
    SUPPLIER_WEBSITE: [value or N/A]
    INVOICE_NUMBER: [Invoice number]
    INVOICE_DATE: [Invoice date]
    PO_NUMBER: [Purchase order number]
    DUE_DATE: [Due date]
    PAYER_NAME: [Payer name]
    PAYER_ADDRESS: [Payer address]
    PAYER_PHONE: [Payer phone]
    SHIP_TO_NAME: [Ship to name]
    SHIP_TO_ADDRESS: [Ship to address]
    SHIP_TO_PHONE: [Ship to phone]
    ITEM_DESCRIPTIONS: [List all item descriptions separated by |]
    ITEM_QUANTITIES: [List all quantities separated by |]
    ITEM_UNIT_PRICES: [List all unit prices separated by |]
    ITEM_TOTALS: [List all item total amounts separated by |]
    SUBTOTAL: [Subtotal amount]
    GST: [GST amount and percentage]
    TOTAL: [Total amount]
    PAYMENT_METHOD: [Payment method details]
    BSB_NUMBER: [6-digit BSB from bank statements only or N/A]
    BANK_ACCOUNT_NUMBER: [account number from bank statements only or N/A]

    Only extract information that is clearly visible in the document. Use "N/A" for any fields not present.