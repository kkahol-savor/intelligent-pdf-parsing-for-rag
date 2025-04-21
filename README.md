# Intelligent PDF Parsing for RAG

A Python-based tool that converts PDF documents to markdown format with intelligent table extraction capabilities using OpenAI's vision API. This tool is designed to enhance RAG (Retrieval-Augmented Generation) systems by providing structured, clean text output from PDF documents.

## Features

- **High-Quality PDF Processing**
  - Converts PDF pages to high-resolution images
  - Preserves original text formatting and structure
  - Handles multi-page documents efficiently

- **Intelligent Table Detection**
  - Two-phase table detection approach
  - Uses OpenAI's vision capabilities for accurate table identification
  - Handles complex table layouts and merged cells
  - Preserves table structure and formatting

- **Context-Aware Image Descriptions**
  - Extracts up to 200 words of surrounding text for each image
  - Provides rich context for more accurate image descriptions
  - Maintains semantic relationships between text and images
  - Particularly effective for figures, tables, and diagrams with captions

- **Comprehensive Output**
  - Generates clean, well-formatted markdown
  - Exports tables as separate CSV files
  - Preserves document structure and hierarchy
  - Maintains semantic relationships between content

- **Robust Error Handling**
  - Graceful handling of API rate limits
  - Automatic retries with exponential backoff
  - Comprehensive error recovery
  - Detailed error logging

- **Easy Integration**
  - Simple API for PDF processing
  - Configurable output formats
  - Flexible directory structure
  - Clear documentation and examples

## Prerequisites

- Python 3.8+
- Azure OpenAI API access
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kkahol-savor/intelligent-pdf-parsing-for-rag.git
cd intelligent-pdf-parsing-for-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following variables:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=your_deployment
IMAGE_DESCRIPTION_PROMPT=your_prompt
TABLE_DETECTION_PROMPT=your_prompt
TABLE_EXTRACTION_PROMPT=your_prompt
```

## Usage

### Basic Usage

```python
from src.pdf_to_markdown import PDFProcessor

# Initialize the processor
processor = PDFProcessor()

# Process a PDF file
markdown_text, table_files = processor.process_pdf("path/to/your/file.pdf")

# Access the results
print(markdown_text)  # The converted markdown text
for table_file in table_files:
    print(f"Table saved to: {table_file}")  # Paths to extracted table CSV files
```

### Advanced Configuration

The `PDFProcessor` class can be customized with different prompts and settings:

```python
processor = PDFProcessor()
processor.image_prompt = "Custom image description prompt"
processor.table_detection_prompt = "Custom table detection prompt"
processor.table_extraction_prompt = "Custom table extraction prompt"
```

## Project Structure

```
intelligent-pdf-parsing-for-rag/
├── src/
│   └── pdf_to_markdown.py    # Main PDF processing logic
├── tests/
│   └── test_pdf_processor.py # Test suite
├── requirements.txt          # Project dependencies
├── .env                     # Environment configuration
└── README.md               # This file
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## ENV SAMPLE PROMPTS

- IMAGE_DESCRIPTION_PROMPT=Describe this image in detail, be specific about the content and context of the image.
- TABLE_DETECTION_PROMPT=Identify if the given image of a page contains a table. The page image is 2480 px by 3508 px. Ignore the outer 80 px on all four sides. If it does, return the bounding box of the table. If it does not, return null. Ignore the outer edges of the page. and only consider tables that are within the inner edges of the page. return the pixel coordinates of the table as a json with the following format: {"x1": 100, "y1": 100, "x2": 200, "y2": 200} where x1, y1 are the top left coordinates and x2, y2 are the bottom right coordinates. add margin of 10 pixels around the table. if it goes out of the page, return the coordinates of the table within the page. if no table is detected, return null. Remember: no prose, only valid JSON.
- TABLE_EXTRACTION_PROMPT=Extract the table content from this image and return it in markdown table format. The format should be:\n| Header1 | Header2 | Header3 |\n|---------|---------|----------|\n| Data1   | Data2   | Data3   |\n\nEnsure to:\n1. Preserve the exact text as shown in the table\n2. Include all headers and data cells\n3. Maintain proper markdown table formatting with | and - characters\n4. Align the columns properly\nReturn ONLY the markdown table, no additional text.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the vision API capabilities
- PyMuPDF for PDF processing functionality
- The open-source community for various tools and libraries used in this project 