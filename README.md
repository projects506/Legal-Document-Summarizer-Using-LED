# Legal Document Summarization

## Overview
This project presents an AI-powered system for summarizing complex legal documents, addressing the challenges posed by dense legal language, intricate structures, and extensive length. Leveraging advanced natural language processing (NLP) techniques, the system condenses lengthy texts into concise, contextually accurate summaries. This tool aids legal professionals, researchers, and other stakeholders in efficiently navigating legal texts.

## Dataset Details
- **Source**: Indian legal materials including Supreme Court case records, Indian Penal Code, Code of Criminal Procedure, and Constitution of India.
- **Size**: 7,130 document-summary pairs; subset of 2,500 pairs used for this study due to hardware constraints.
- **Preprocessing**:
  - Converted PDF files to text.
  - Tokenized texts and summaries using LEDTokenizer.
  - Split into training (80%), validation (10%), and testing (10%) subsets.

## Methodology
The project employs a fine-tuned Longformer Encoder-Decoder (LED) model optimized for lengthy texts and tailored to Indian legal terminology. Key techniques include:
- **Beam Search**: Ensures optimal summary generation with a beam width of 4.
- **Knowledge Representation**:
  - Positional embeddings for document structure.
  - Attention mechanisms for preserving legal citations and argumentative flow.
  - Domain-specific ontologies to encapsulate Indian legal terminology.
- **Constraint Satisfaction**: Enforces structural and contextual constraints, maintaining coherence and legal integrity.

## Key Features
- **Accuracy**: Extracts and preserves critical legal arguments, citations, and rulings.
- **Efficiency**: Summarizes multi-page documents in seconds.
- **Clarity**: Generates summaries that are easily understandable, even for non-legal users.
- **Versatility**: Adapts to various document types, including case laws, contracts, and statutes.

## Results
- Demonstrated improvements in summarization clarity, relevance, and efficiency.
- Enhanced workflow efficiency for legal professionals by reducing manual review time.
- Reliable performance across different legal document types and complexities.

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/project506/Legal-Document-Summarizer-Using-LED.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Summarization Script**:
   ```bash
   python app.py
   ```

## Dependencies
- `transformers`
- `torch`
- `pandas`
- `numpy`

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
