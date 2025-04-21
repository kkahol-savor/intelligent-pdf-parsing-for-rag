"""Test script for the PDF processor functionality."""

from pdf_to_markdown import PDFProcessor
import os
from pathlib import Path
import time

def main():
    """Run a test of the PDF processor on the sample file."""
    # Initialize the processor
    processor = PDFProcessor()
    
    # Get the path to the sample PDF
    pdf_path = Path("data/sample_file.pdf")
    
    if not pdf_path.exists():
        print(f"Error: Sample PDF file not found at {pdf_path}")
        return
    
    print(f"Processing PDF file: {pdf_path}")
    
    try:
        # Process the PDF with a delay between API calls
        print("Starting PDF processing...")
        markdown_text, table_files = processor.process_pdf(pdf_path)
        
        # Print the first 500 characters of the markdown text
        print("\nFirst 500 characters of the markdown output:")
        print("-" * 80)
        print(markdown_text[:500])
        print("-" * 80)
        
        # Print information about extracted tables
        if table_files:
            print("\nExtracted tables:")
            for table_file in table_files:
                print(f"- {table_file}")
        else:
            print("\nNo tables were extracted from the PDF.")
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if your Azure OpenAI API key is valid")
        print("2. Verify that you have sufficient quota available")
        print("3. Try processing a smaller PDF file")
        print("4. Add more delay between API calls")

if __name__ == "__main__":
    main() 