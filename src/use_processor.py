#!/usr/bin/env python3
"""
Example script demonstrating how to use the PDFProcessor class.
This script shows how to process a PDF file and print table bounding boxes.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from pdf_to_markdown import PDFProcessor
import fitz

def main():
    """Main function to demonstrate PDFProcessor usage."""
    # Load environment variables
    load_dotenv(override=True)
    
    # Print environment variables
    print("Environment Variables:")
    print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"AZURE_OPENAI_DEPLOYMENT: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
    print(f"TABLE_DETECTION_PROMPT: {os.getenv('TABLE_DETECTION_PROMPT')}")
    print(f"TABLE_EXTRACTION_PROMPT: {os.getenv('TABLE_EXTRACTION_PROMPT')}")
    print(f"IMAGE_DESCRIPTION_PROMPT: {os.getenv('IMAGE_DESCRIPTION_PROMPT')}")
    print()
    
    # Check if a PDF file path was provided
    if len(sys.argv) < 2:
        print("Usage: python use_processor.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)
    
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize the processor
    processor = PDFProcessor()
    
    # Process the PDF
    try:
        # Override the _detect_table method to add more logging for page 4
        original_detect_table = processor._detect_table
        
        def enhanced_detect_table(image_path, page_num=None):
            """Enhanced version of _detect_table with additional logging."""
            print(f"\nDetecting tables on page {page_num}...")
            result = original_detect_table(image_path, page_num)
            
            if page_num == 4:  # Special handling for page 4
                print(f"\n=== DETAILED LOGGING FOR PAGE 4 ===")
                print(f"Image path: {image_path}")
                print(f"Table detection result: {result}")
                
                # Check if the image exists and its size
                if os.path.exists(image_path):
                    from PIL import Image
                    img = Image.open(image_path)
                    print(f"Image dimensions: {img.width}x{img.height} pixels")
                    print(f"Image mode: {img.mode}")
                    img.close()
                else:
                    print(f"Image file not found: {image_path}")
                
                # If no table detected, try to analyze the image content
                if not result:
                    print("No table detected on page 4. Analyzing image content...")
                    # You could add additional analysis here if needed
                
                print(f"=== END OF PAGE 4 LOGGING ===\n")
            
            return result
        
        # Replace the _detect_table method with our enhanced version
        processor._detect_table = enhanced_detect_table
        
        # Override the _save_page_image method to fix the page indexing issue
        original_save_page_image = processor._save_page_image
        
        def fixed_save_page_image(page, page_num):
            """Fixed version of _save_page_image to ensure correct page indexing."""
            # Ensure output directory exists
            processor.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the correct page number for the filename
            image_path = processor.output_dir / f"page_{page_num}.png"
            
            # Save the page as an image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            pix.save(str(image_path))
            
            print(f"Saved page {page_num} image to: {image_path}")
            return image_path
        
        # Replace the _save_page_image method with our fixed version
        processor._save_page_image = fixed_save_page_image
        
        # Process the PDF
        markdown_text, table_files = processor.process_pdf(pdf_path)
        
        # Print summary
        print("\nProcessing complete!")
        print(f"Generated markdown text length: {len(markdown_text)} characters")
        print(f"Extracted {len(table_files)} tables")
        
        # Print paths to extracted table files
        if table_files:
            print("\nExtracted table files:")
            for table_file in table_files:
                print(f"  - {table_file}")
        else:
            print("\nNo tables were extracted from the PDF.")
        
        # Save markdown to file
        output_file = Path(pdf_path).with_suffix('.md')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"\nMarkdown saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 