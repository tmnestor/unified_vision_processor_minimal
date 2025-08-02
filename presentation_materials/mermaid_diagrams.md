# Mermaid Diagrams for Vision Transformer Presentation

This file contains all the Mermaid diagram definitions for the presentation. These can be embedded directly into markdown files or rendered separately.

## 1. Vision Transformer Architecture

```mermaid
graph TB
    subgraph "Vision Transformer Architecture"
        A[Input Image<br/>224x224] --> B[Patch Division<br/>16x16 patches]
        B --> C[Linear Projection<br/>Patch Embeddings]
        C --> D[Add Position<br/>Embeddings]
        D --> E[Transformer<br/>Encoder Blocks]
        E --> F[Multi-Head<br/>Self-Attention]
        E --> G[Feed Forward<br/>Network]
        F --> H[Layer Norm]
        G --> H
        H --> I[Output<br/>Features]
    end
    
    style A fill:#e8f4fd
    style B fill:#b8e0d2
    style C fill:#b8e0d2
    style D fill:#b8e0d2
    style E fill:#d6eadf
    style F fill:#d6eadf
    style G fill:#d6eadf
    style I fill:#eac4d5
```

## 2. LayoutLM vs Vision Transformer Comparison

```mermaid
graph TB
    subgraph "LayoutLM Pipeline"
        L1[Document Image] --> L2[OCR Engine]
        L1 --> L3[R-CNN Features]
        L1 --> L4[Layout Coordinates]
        L2 --> L5[Text + Boxes]
        L3 --> L6[Image Features]
        L4 --> L7[2D Positions]
        L5 --> L8[LayoutLM<br/>Transformer]
        L6 --> L8
        L7 --> L8
        L8 --> L9[Extracted Fields]
        
        L2 -.-> LX1[❌ OCR Failures]
        L8 -.-> LX2[⚠️ Alignment Issues]
    end
    
    subgraph "Vision Transformer Pipeline"
        V1[Document Image] --> V2[Vision Transformer<br/>+ Language Model]
        V2 --> V3[Extracted Fields]
        V2 -.-> VX1[✅ End-to-End]
    end
    
    style L2 fill:#ffe4e1
    style L3 fill:#e0e0ff
    style L4 fill:#e0ffe0
    style L8 fill:#ffe4ff
    style LX1 fill:#ff6b6b,color:#fff
    style LX2 fill:#ffa500,color:#fff
    
    style V2 fill:#90ee90
    style VX1 fill:#32cd32,color:#fff
```

## 3. Self-Attention Mechanism

```mermaid
graph LR
    subgraph "Self-Attention for Document Understanding"
        P1[Patch 1<br/>Header] -.->|0.9| P5[Patch 5<br/>Total]
        P2[Patch 2<br/>Items] -.->|0.7| P5
        P3[Patch 3<br/>Prices] -.->|0.8| P5
        P4[Patch 4<br/>Tax] -.->|0.6| P5
        P5 -.->|0.3| P6[Patch 6<br/>Footer]
        
        P1 -.->|0.2| P2
        P2 -.->|0.9| P3
        P3 -.->|0.7| P4
    end
    
    style P1 fill:#ffb6c1
    style P2 fill:#98fb98
    style P3 fill:#87ceeb
    style P4 fill:#dda0dd
    style P5 fill:#ffd700
    style P6 fill:#f0e68c
```

## 4. Document Processing Pipeline Comparison

```mermaid
graph LR
    subgraph "Traditional OCR Pipeline"
        T1[Image] --> T2[OCR]
        T2 --> T3[Text]
        T3 --> T4[Rules/<br/>Regex]
        T4 --> T5[Fields]
        
        T2 -.-> TX1[OCR Errors]
        T3 -.-> TX2[Layout Lost]
        T4 -.-> TX3[Rule Failures]
    end
    
    subgraph "Vision Transformer Pipeline"
        V1[Image] --> V2[ViT Model]
        V2 --> V3[Fields]
        
        V2 -.-> VX1[Direct Understanding]
    end
    
    style T2 fill:#ffd700
    style T3 fill:#98fb98
    style T4 fill:#87ceeb
    style TX1 fill:#ff6b6b,color:#fff
    style TX2 fill:#ff6b6b,color:#fff
    style TX3 fill:#ff6b6b,color:#fff
    
    style V2 fill:#90ee90
    style VX1 fill:#32cd32,color:#fff
```


## 5. Vision Transformer Components Detail

```mermaid
graph TD
    subgraph "Patch Embedding Process"
        I1[Input Image<br/>224x224x3] --> I2[Divide into<br/>14x14 Patches]
        I2 --> I3[16x16 pixels<br/>per patch]
        I3 --> I4[Flatten:<br/>196 patches]
        I4 --> I5[Linear<br/>Projection]
        I5 --> I6[Add Position<br/>Embeddings]
    end
    
    subgraph "Transformer Block"
        T1[Input] --> T2[Layer Norm]
        T2 --> T3[Multi-Head<br/>Attention]
        T3 --> T4[Residual<br/>Connection]
        T4 --> T5[Layer Norm]
        T5 --> T6[MLP]
        T6 --> T7[Residual<br/>Connection]
        T7 --> T8[Output]
    end
    
    I6 --> T1
    
    style I1 fill:#e8f4fd
    style I5 fill:#b8e0d2
    style I6 fill:#b8e0d2
    style T3 fill:#ffd700
    style T6 fill:#98fb98
```

## 6. Performance Comparison

```mermaid
graph TB
    subgraph "Metrics Comparison"
        M1[Success Rate] --> M2[LayoutLM: 85%<br/>ViT: 100%]
        M3[Field Accuracy] --> M4[LayoutLM: 47%<br/>ViT: 59%]
        M5[Pipeline Steps] --> M6[LayoutLM: 3+<br/>ViT: 1]
        M7[Memory Usage] --> M8[LayoutLM: 8GB<br/>ViT: 2.6GB]
    end
    
    style M2 fill:#90ee90
    style M4 fill:#90ee90
    style M6 fill:#90ee90
    style M8 fill:#90ee90
```

## 7. Vision Transformer Detailed Pipeline

```mermaid
graph TD
    subgraph "Input Processing"
        A[Document Image<br/>448x448] --> B[Patch Division<br/>32x32 patches]
        B --> C[14x14 patches<br/>each 32x32 pixels]
        C --> D[Flatten to<br/>196 patch vectors]
    end
    
    subgraph "Embedding Layer"
        D --> E[Linear Projection<br/>768-dim embeddings]
        E --> F[Add Positional<br/>Embeddings]
        F --> G[Add [CLS] Token]
    end
    
    subgraph "Transformer Blocks (x12)"
        G --> H[Multi-Head<br/>Self-Attention]
        H --> I[Add & Norm]
        I --> J[Feed Forward<br/>Network]
        J --> K[Add & Norm]
        K --> L[Next Block]
    end
    
    subgraph "Language Model Head"
        L --> M[Vision Encoder<br/>Output]
        M --> N[Language Model<br/>Decoder]
        N --> O[Text Generation<br/>KEY: VALUE]
    end
    
    style A fill:#e8f4fd
    style E fill:#b8e0d2
    style F fill:#b8e0d2
    style H fill:#ffd700
    style J fill:#98fb98
    style N fill:#dda0dd
    style O fill:#eac4d5
```

## 8. Image Patch Processing Detail

```mermaid
graph LR
    subgraph "Single Document Patch Processing"
        P1[Patch: Header<br/>32x32 pixels] --> P2[Flatten:<br/>3072 values]
        P2 --> P3[Linear Layer:<br/>768 dims]
        P3 --> P4[+ Position<br/>Embedding]
        P4 --> P5[Patch Token<br/>Ready for Attention]
    end
    
    subgraph "Attention Mechanism"
        P5 --> A1[Query Q]
        P5 --> A2[Key K] 
        P5 --> A3[Value V]
        A1 --> A4[Attention<br/>Weights]
        A2 --> A4
        A4 --> A5[Weighted<br/>Values]
        A3 --> A5
        A5 --> A6[Output<br/>Representation]
    end
    
    subgraph "Global Context"
        A6 --> G1[Attend to<br/>ALL patches]
        G1 --> G2[Header ↔ Total<br/>Items ↔ Prices<br/>Logo ↔ Company]
        G2 --> G3[Rich Context<br/>Understanding]
    end
    
    style P1 fill:#ffb6c1
    style P3 fill:#b8e0d2
    style A4 fill:#ffd700
    style G2 fill:#98fb98
    style G3 fill:#eac4d5
```

## 9. Multi-Head Attention Detail

```mermaid
graph TB
    subgraph "Multi-Head Self-Attention (8 heads)"
        I[Input Patches<br/>196 x 768] --> Q[Query Matrix<br/>Q = X · W_q]
        I --> K[Key Matrix<br/>K = X · W_k]
        I --> V[Value Matrix<br/>V = X · W_v]
        
        Q --> H1[Head 1<br/>Q₁K₁ᵀV₁]
        Q --> H2[Head 2<br/>Q₂K₂ᵀV₂]
        Q --> H3[Head 3<br/>Q₃K₃ᵀV₃]
        Q --> H8[... Head 8<br/>Q₈K₈ᵀV₈]
        
        K --> H1
        K --> H2  
        K --> H3
        K --> H8
        
        V --> H1
        V --> H2
        V --> H3
        V --> H8
        
        H1 --> C[Concatenate<br/>All Heads]
        H2 --> C
        H3 --> C
        H8 --> C
        
        C --> O[Output Projection<br/>W_o]
        O --> R[Final Output<br/>196 x 768]
    end
    
    style I fill:#e8f4fd
    style Q fill:#b8e0d2
    style K fill:#b8e0d2  
    style V fill:#b8e0d2
    style H1 fill:#ffd700
    style H2 fill:#ffd700
    style H3 fill:#ffd700
    style H8 fill:#ffd700
    style C fill:#98fb98
    style R fill:#eac4d5
```

## 10. Document Understanding Flow

```mermaid
graph LR
    subgraph "Document Patch Analysis"
        D1[Invoice Header<br/>Patch 1-20] --> A1[Attention Layer 1:<br/>Local features]
        D2[Line Items<br/>Patch 21-80] --> A1
        D3[Total Section<br/>Patch 81-120] --> A1
        D4[Footer Info<br/>Patch 121-196] --> A1
        
        A1 --> A2[Attention Layer 6:<br/>Regional relationships]
        A2 --> A3[Attention Layer 12:<br/>Global understanding]
    end
    
    subgraph "Semantic Understanding"
        A3 --> S1[Header ↔ Company Info]
        A3 --> S2[Items ↔ Quantities]
        A3 --> S3[Prices ↔ Total]
        A3 --> S4[Date ↔ Due Date]
        
        S1 --> LM[Language Model<br/>Generation]
        S2 --> LM
        S3 --> LM
        S4 --> LM
    end
    
    subgraph "Output Generation"
        LM --> O1[SUPPLIER: Hyatt Hotels]
        LM --> O2[TOTAL: $521.21]
        LM --> O3[GST: $47.38]
        LM --> O4[INVOICE_DATE: 2024-03-15]
    end
    
    style D1 fill:#ffb6c1
    style D2 fill:#98fb98
    style D3 fill:#87ceeb
    style D4 fill:#dda0dd
    style A3 fill:#ffd700
    style LM fill:#eac4d5
    style O1 fill:#f0e68c
    style O2 fill:#f0e68c
    style O3 fill:#f0e68c
    style O4 fill:#f0e68c
```

## Usage Notes

### Embedding in Markdown
```markdown
```mermaid
[paste diagram code here]
```
```

### Rendering Options
1. **GitHub/GitLab**: Renders automatically in markdown files
2. **VS Code**: Install Markdown Preview Mermaid Support extension
3. **Presentations**: Many tools (Reveal.js, Marp) support Mermaid
4. **Export**: Use Mermaid CLI or online editor to export as PNG/SVG

### Customization
- Colors can be adjusted in the `style` statements
- Layout direction: TB (top-bottom), LR (left-right), TD (top-down)
- Shapes: Rectangle (default), Round ([]), Diamond {{}}, Circle (())