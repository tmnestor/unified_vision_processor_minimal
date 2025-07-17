# Ultra-simple classification prompt that bypasses Llama safety mode
ULTRA_SIMPLE_PROMPT = """<|image|>What type of document is this?

Choose one: TAX_INVOICE, BUSINESS_RECEIPT, FUEL_RECEIPT, OTHER_BUSINESS

Answer: """

# Alternative prompts to try if first fails
ALTERNATIVE_PROMPTS = [
    "<|image|>Is this a receipt or invoice? Answer: TAX_INVOICE or BUSINESS_RECEIPT",
    "<|image|>Document type?",
    "<|image|>Receipt type: FUEL_RECEIPT, BUSINESS_RECEIPT, or OTHER_BUSINESS?",
]

print("Copy these simplified prompts to test:")
print("1. Ultra-simple:", ULTRA_SIMPLE_PROMPT)
print("\n2. Alternatives:")
for i, prompt in enumerate(ALTERNATIVE_PROMPTS, 1):
    print(f"   {i}. {prompt}")
