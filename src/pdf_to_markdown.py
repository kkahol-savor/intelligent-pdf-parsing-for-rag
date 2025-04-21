"""PDF to Markdown converter with intelligent table extraction capabilities.

This module provides a robust solution for converting PDF documents to markdown format,
with special emphasis on table extraction using OpenAI's vision capabilities. It is
specifically designed to enhance RAG (Retrieval-Augmented Generation) systems by
providing structured, clean text output from PDF documents.

Key Features:
- PDF to Markdown conversion with preserved formatting
- Intelligent table detection using OpenAI's vision API
- Two-phase table extraction process
- Automatic CSV export for tabular data
- Rate limit handling with exponential backoff
- Comprehensive error handling and logging

The module uses a combination of PyMuPDF for PDF processing and OpenAI's vision
capabilities for table detection and extraction, making it particularly effective
for documents with complex layouts and tabular data.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import fitz  # PyMuPDF
import base64
import requests
from pathlib import Path
import json
import re
import csv
import io
import time
from PIL import Image

class PDFProcessor:
    """A class to process PDF files and convert them to markdown format with intelligent table extraction.

    This class provides comprehensive functionality for processing PDF documents,
    with special attention to table detection and extraction. It uses OpenAI's
    vision capabilities to identify and extract tables from PDF pages, making it
    particularly effective for documents with complex layouts.

    The processor implements a two-phase approach for table handling:
    1. Table Detection: Uses OpenAI's vision API to identify table regions
    2. Table Extraction: Extracts and formats table content using vision API

    Features:
    - High-resolution page rendering
    - Intelligent table detection
    - Accurate table structure preservation
    - Automatic CSV export
    - Rate limit handling
    - Comprehensive error recovery
    - Context-aware image descriptions (up to 200 words before and after images)

    The context-aware image description feature:
    - Extracts up to 200 words of text before and after each image
    - Provides this context to the vision API for more accurate descriptions
    - Helps maintain semantic relationships between text and images
    - Particularly useful for figures, tables, and diagrams with captions

    Attributes:
        endpoint (str): Azure OpenAI API endpoint
        api_key (str): Azure OpenAI API key
        deployment (str): Azure OpenAI deployment name
        image_prompt (str): Prompt for image description
        table_detection_prompt (str): Prompt for table detection
        table_extraction_prompt (str): Prompt for table extraction
        output_dir (Path): Directory for output images
        tables_dir (Path): Directory for extracted tables
        table_images_dir (Path): Directory for table images
        max_retries (int): Maximum number of retries for API calls
        base_delay (int): Base delay for exponential backoff
        api_delay (int): Delay between API calls in seconds

    Example:
        >>> processor = PDFProcessor()
        >>> markdown_text, table_files = processor.process_pdf("document.pdf")
        >>> print(markdown_text)  # The converted markdown text
        >>> for table_file in table_files:
        ...     print(f"Table saved to: {table_file}")
    """

    def __init__(self) -> None:
        """Initialize the PDFProcessor with configuration from environment variables."""
        load_dotenv(override=True)
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.image_prompt = os.getenv("IMAGE_DESCRIPTION_PROMPT", "Describe this image in detail")
        self.table_detection_prompt = os.getenv("TABLE_DETECTION_PROMPT", 
            "Identify if the given image of a page contains a table. If it does, return the bounding box of the table. "
            "If it does not, return null. Ignore the outer edges of the page. and only consider tables that are within "
            "the inner edges of the page. return the pixel coordinates of the table as a json with the following format: "
            '{"x1": 100, "y1": 100, "x2": 200, "y2": 200} where x1, y1 are the top left coordinates and x2, y2 are the '
            "bottom right coordinates. if no table is detected, return null.")
        self.table_extraction_prompt = os.getenv("TABLE_EXTRACTION_PROMPT",
            "Extract the table content from this image and return it in markdown table format. "
            "The format should be:\n"
            "| Header1 | Header2 | Header3 |\n"
            "|---------|---------|----------|\n"
            "| Data1   | Data2   | Data3   |\n\n"
            "Ensure to:\n"
            "1. Preserve the exact text as shown in the table\n"
            "2. Include all headers and data cells\n"
            "3. Maintain proper markdown table formatting with | and - characters\n"
            "4. Align the columns properly\n"
            "Return ONLY the markdown table, no additional text.")
        
        # Create output directories
        self.output_dir = Path("output_images")
        self.output_dir.mkdir(exist_ok=True)
        self.tables_dir = Path("tables")
        self.tables_dir.mkdir(exist_ok=True)
        self.table_images_dir = self.tables_dir / "images"
        self.table_images_dir.mkdir(exist_ok=True)
        
        self.max_retries = 5
        self.base_delay = 2  # Increased base delay in seconds
        self.api_delay = 3  # Delay between API calls in seconds

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode an image file to base64 string.

        This method reads an image file and encodes it as a base64 string,
        which is required for sending images to the OpenAI vision API.

        Args:
            image_path (Union[str, Path]): Path to the image file to encode

        Returns:
            str: Base64 encoded string representation of the image

        Raises:
            FileNotFoundError: If the image file does not exist
            IOError: If there are issues reading the image file
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_surrounding_text(self, page, image_rect, word_limit=200):
        """Get text before and after an image within a word limit.

        This method extracts text blocks that appear before and after an image
        on a PDF page, up to a specified word limit. This context is useful
        for providing better image descriptions.

        Args:
            page: PyMuPDF page object
            image_rect: Rectangle defining the image boundaries
            word_limit (int, optional): Maximum number of words to extract. Defaults to 200.

        Returns:
            Tuple[str, str]: A tuple containing (text_before_image, text_after_image)
        """
        # Get all text blocks on the page
        blocks = page.get_text("blocks")
        before_text = []
        after_text = []
        word_count = 0
        
        # Sort blocks by vertical position
        blocks.sort(key=lambda b: b[1])  # Sort by y-coordinate
        
        for block in blocks:
            block_rect = fitz.Rect(block[:4])
            block_text = block[4]
            
            # If block is above the image
            if block_rect.y1 <= image_rect.y0:
                before_text.append(block_text)
                word_count += len(block_text.split())
                if word_count >= word_limit:
                    break
        
        # Reset word count for after text
        word_count = 0
        
        # Get text after the image
        for block in blocks:
            block_rect = fitz.Rect(block[:4])
            block_text = block[4]
            
            # If block is below the image
            if block_rect.y0 >= image_rect.y1:
                after_text.append(block_text)
                word_count += len(block_text.split())
                if word_count >= word_limit:
                    break
        
        return " ".join(before_text), " ".join(after_text)

    def _get_image_description(self, image_path, context_before="", context_after=""):
        """Get a description of an image using OpenAI's vision API.

        This method sends an image to OpenAI's vision API along with optional
        context text that appears before and after the image. The API returns
        a natural language description of the image.

        Args:
            image_path (Union[str, Path]): Path to the image file
            context_before (str, optional): Text that appears before the image. Defaults to "".
            context_after (str, optional): Text that appears after the image. Defaults to "".

        Returns:
            str: Natural language description of the image

        Raises:
            Exception: If the API call fails or returns an error
        """
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

        base64_image = self._encode_image(image_path)
        
        # Create context-aware prompt
        context_prompt = f"{self.image_prompt}\n\nContext before the image:\n{context_before}\n\nContext after the image:\n{context_after}"
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": context_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500  # Increased token limit for longer descriptions
        }

        response = requests.post(
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-15-preview",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Error getting image description: {response.status_code}"

    def _page_to_image(self, page, zoom=2):
        """Convert a PDF page to an image.

        This method renders a PDF page as a high-resolution image,
        which is necessary for table detection and extraction.

        Args:
            page: PyMuPDF page object
            zoom (int, optional): Zoom factor for higher resolution. Defaults to 2.

        Returns:
            bytes: PNG image data as bytes
        """
        # Get the page's pixmap with higher resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        return img_bytes

    def _call_openai_with_retry(self, payload, operation_name):
        """Make OpenAI API call with exponential backoff retry logic.

        This method implements a robust retry mechanism for OpenAI API calls,
        handling rate limits and temporary failures with exponential backoff.

        Args:
            payload (dict): The API request payload
            operation_name (str): Name of the operation for logging purposes

        Returns:
            dict: API response data if successful, None otherwise

        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-15-preview",
                    headers={
                        "api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', self.base_delay * (2 ** attempt)))
                    print(f"Rate limit hit for {operation_name}. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                else:
                    print(f"Error in {operation_name}: {response.status_code}")
                    print(response.text)
                    return None
                    
            except Exception as e:
                print(f"Exception in {operation_name}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2 ** attempt))
                    continue
                return None
        
        return None

    def _save_table_image(self, page, bbox, page_num, table_num):
        """Save the table region as an image.

        This method extracts a specific region of a PDF page (defined by a
        bounding box) and saves it as a separate image file.

        Args:
            page: PyMuPDF page object
            bbox (dict): Bounding box coordinates {x1, y1, x2, y2}
            page_num (int): Page number (0-based)
            table_num (int): Table number on the page (0-based)

        Returns:
            Path: Path to the saved table image
        """
        # Create a new pixmap for the table region
        table_rect = fitz.Rect(bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=table_rect)
        
        # Save the image
        image_path = self.table_images_dir / f"page_{page_num + 1}_table_{table_num + 1}.png"
        pix.save(str(image_path))
        return image_path

    def _extract_table_from_image(self, image_path):
        """Extract table content from an image using OpenAI.

        This method sends a table image to OpenAI's vision API to extract
        its content and structure. The API returns the table in markdown format.

        Args:
            image_path (Union[str, Path]): Path to the table image

        Returns:
            str: Table content in markdown format, or None if extraction fails

        Raises:
            Exception: If the API call fails or returns an error
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Prepare the API call
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.table_extraction_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            # Make the API call with retry logic
            response = self._call_openai_with_retry(payload, "table extraction")
            if not response:
                return None
            
            result = response['choices'][0]['message']['content']
            print(f"\nTable extraction response for {image_path}:")
            print(result)
            
            # Since we're getting markdown directly, just return it
            return result.strip()
                
        except Exception as e:
            print(f"Error extracting table from image: {e}")
            return None

    def _detect_tables_with_openai(self, pdf_path):
        """Use OpenAI to detect tables in each page by analyzing page images.

        This method processes each page of a PDF document, converting it to
        an image and using OpenAI's vision API to detect tables. It returns
        information about all detected tables, including their page numbers
        and bounding box coordinates.

        Args:
            pdf_path (Union[str, Path]): Path to the PDF file

        Returns:
            List[dict]: List of detected tables with page numbers and bounding boxes

        Raises:
            Exception: If there are issues processing the PDF or calling the API
        """
        try:
            doc = fitz.open(pdf_path)
            pages_with_tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                img_bytes = self._page_to_image(page)
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.table_detection_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 100
                }
                
                response = self._call_openai_with_retry(payload, "table detection")
                if not response:
                    continue
                
                result = response['choices'][0]['message']['content']
                print(f"\nPage {page_num + 1} OpenAI Response:")
                print(result)
                
                try:
                    table_info = json.loads(result)
                    if table_info is not None:
                        pages_with_tables.append({
                            'page_num': page_num + 1,
                            'bbox': table_info
                        })
                except json.JSONDecodeError as e:
                    print(f"Error parsing table detection response for page {page_num + 1}: {e}")
                    print(f"Raw response: {result}")
            
            return pages_with_tables
            
        except Exception as e:
            print(f"Error detecting tables with OpenAI: {e}")
            return []

    def _extract_tables(self, page, bbox=None):
        """Extract tables from a page using text block analysis.

        This method analyzes text blocks on a PDF page to identify and extract
        tables. It can optionally focus on a specific region defined by a
        bounding box.

        Args:
            page: PyMuPDF page object
            bbox (dict, optional): Bounding box to limit the search area. Defaults to None.

        Returns:
            List[str]: List of extracted tables in markdown format
        """
        # Get all text blocks
        blocks = page.get_text("blocks")
        
        # Sort blocks by vertical position
        blocks.sort(key=lambda b: b[1])
        
        tables = []
        current_table = []
        
        for block in blocks:
            block_text = block[4]
            block_rect = fitz.Rect(block[:4])
            
            # Skip empty blocks
            if not block_text.strip():
                continue
            
            # If bbox is provided, only process blocks within it
            if bbox:
                if not (bbox['x1'] <= block_rect.x0 <= bbox['x2'] and 
                       bbox['y1'] <= block_rect.y0 <= bbox['y2']):
                    continue
                
            # Check if this block might be part of a table
            # Look for patterns like multiple spaces, tabs, or consistent spacing
            if re.search(r'\s{2,}', block_text) or '\t' in block_text:
                # Split by whitespace or tabs
                cells = re.split(r'\s{2,}|\t', block_text)
                
                # If we have multiple cells, this might be a table row
                if len(cells) > 1:
                    # Clean up cells
                    cells = [cell.strip() for cell in cells if cell.strip()]
                    if cells:  # Only add non-empty rows
                        current_table.append(cells)
                else:
                    # If we were building a table and this block doesn't fit,
                    # save the current table and start a new one
                    if current_table:
                        tables.append(current_table)
                        current_table = []
            else:
                # If we were building a table and this block doesn't fit,
                # save the current table and start a new one
                if current_table:
                    tables.append(current_table)
                    current_table = []
        
        # Add the last table if there is one
        if current_table:
            tables.append(current_table)
        
        # Convert tables to markdown
        markdown_tables = []
        for table in tables:
            if len(table) >= 2:  # Need at least header and one data row
                markdown_table = self._convert_table_to_markdown(table)
                if markdown_table:
                    markdown_tables.append(markdown_table)
        
        return markdown_tables

    def _convert_table_to_markdown(self, table):
        """Convert a table to markdown format.

        This method takes a table represented as a list of rows (where each
        row is a list of cells) and converts it to a properly formatted
        markdown table string.

        Args:
            table (List[List[str]]): Table data as a list of rows

        Returns:
            str: Table in markdown format
        """
        if not table or len(table) < 2:
            return ""
        
        # Get the number of columns (use the maximum number of cells in any row)
        cols = max(len(row) for row in table)
        
        # Create the markdown table header
        header = list(table[0])  # Convert to list if it's not already
        # Pad with empty cells if necessary
        header.extend([""] * (cols - len(header)))
        markdown = "| " + " | ".join([str(cell).strip() for cell in header]) + " |\n"
        
        # Add the separator row
        markdown += "| " + " | ".join(["---"] * cols) + " |\n"
        
        # Add the data rows
        for row in table[1:]:
            # Convert row to list and pad with empty cells if necessary
            row_list = list(row)
            row_list.extend([""] * (cols - len(row_list)))
            markdown += "| " + " | ".join([str(cell).strip() for cell in row_list]) + " |\n"
        
        return markdown

    def _save_table_as_csv(self, table_text: str, page_num: int, table_num: int) -> Path:
        """Save extracted table as CSV file.

        This method takes a table in markdown format and saves it as a CSV file,
        preserving the structure and data.

        Args:
            table_text (str): Table text in markdown format
            page_num (int): Page number (0-based)
            table_num (int): Table number on the page (0-based)

        Returns:
            Path: Path to the saved CSV file, or None if saving fails
        """
        lines = table_text.strip().split('\n')
        if len(lines) < 3:
            return None

        # Parse markdown table
        headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
        data = []
        for line in lines[2:]:
            row = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(row) == len(headers):
                data.append(row)

        # Ensure tables directory exists
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_path = self.tables_dir / f"table_page_{page_num}_{table_num}.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data)

        return csv_path

    def _markdown_to_csv(self, markdown_table):
        """Convert markdown table to CSV format.

        This method parses a markdown table string and converts it to a list
        of rows suitable for CSV writing.

        Args:
            markdown_table (str): Table in markdown format

        Returns:
            List[List[str]]: Table data as a list of rows, or None if conversion fails
        """
        try:
            # Split the markdown table into lines
            lines = markdown_table.strip().split('\n')
            
            # Remove the separator line (the one with dashes)
            lines = [line for line in lines if not line.strip().startswith('|-')]
            
            # Convert each line to CSV format
            csv_data = []
            for line in lines:
                # Remove leading/trailing |
                line = line.strip('|')
                # Split by | and strip whitespace
                row = [cell.strip() for cell in line.split('|')]
                csv_data.append(row)
            
            return csv_data
            
        except Exception as e:
            print(f"Error converting markdown to CSV: {e}")
            return None

    def _call_vision_api(self, image_path: Union[str, Path], prompt: str) -> str:
        """Call the Azure OpenAI Vision API with an image and prompt.

        This method sends an image to the Azure OpenAI Vision API along with
        a prompt, and returns the API's response. It includes retry logic for
        handling rate limits and temporary failures.

        Args:
            image_path (Union[str, Path]): Path to the image file
            prompt (str): Prompt to send with the image

        Returns:
            str: API response text

        Raises:
            Exception: If the API call fails after all retries
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        for attempt in range(self.max_retries):
            try:
                # Add delay between API calls
                if attempt > 0:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                
                response = requests.post(
                    f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-15-preview",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 429:  # Rate limit
                    print(f"Rate limit hit. Attempt {attempt + 1}/{self.max_retries}")
                    if attempt == self.max_retries - 1:
                        raise Exception("Failed to call Vision API: Rate limit exceeded")
                    continue
                elif response.status_code != 200:
                    raise Exception(f"Failed to call Vision API: HTTP {response.status_code}")
                    
                return response.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to call Vision API: {str(e)}")
                print(f"API call failed: {str(e)}")
                
        raise Exception("Failed to call Vision API after all retries")

    def _save_page_image(self, page: fitz.Page, page_num: int) -> Path:
        """Save a PDF page as an image.

        This method renders a PDF page as a high-resolution image and saves
        it to the output directory.

        Args:
            page (fitz.Page): PyMuPDF page object
            page_num (int): Page number (0-based)

        Returns:
            Path: Path to the saved image
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use 1-based page numbering for the filename
        image_path = self.output_dir / f"page_{page_num + 1}.png"
        
        # Save the page as an image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        pix.save(str(image_path))
        
        print(f"Saved page {page_num + 1} image to: {image_path}")
        return image_path

    def _detect_table(self, image_path: Union[str, Path], page_num: Optional[int] = None) -> Optional[Dict[str, int]]:
        """Detect if an image contains a table and return its bounding box.

        This method sends an image to OpenAI's vision API to detect if it
        contains a table, and if so, returns the table's bounding box coordinates.

        Args:
            image_path (Union[str, Path]): Path to the image file
            page_num (Optional[int]): Page number for logging purposes

        Returns:
            Optional[Dict[str, int]]: Dictionary containing table coordinates
                (x1, y1, x2, y2) or None if no table is detected
        """
        response = self._call_vision_api(image_path, self.table_detection_prompt)
        try:
            result = json.loads(response)
            if result and isinstance(result, dict) and 'x1' in result and 'y1' in result and 'x2' in result and 'y2' in result:
                print(f"\nTable detected! Bounding box coordinates:")
                print(f"  Top-left: ({result['x1']}, {result['y1']})")
                print(f"  Bottom-right: ({result['x2']}, {result['y2']})")
                print(f"  Width: {result['x2'] - result['x1']}px")
                print(f"  Height: {result['y2'] - result['y1']}px")
                return result
            else:
                print(f"No table detected in the response: {response}")
                return None
        except json.JSONDecodeError:
            print(f"Error parsing JSON response: {response}")
            return None
        except Exception as e:
            print(f"Error processing table detection response: {e}")
            print(f"Raw response: {response}")
            return None

    def _extract_table(self, image_path: Union[str, Path], bbox: Dict[str, int]) -> str:
        """Extract table content from an image using the provided bounding box.

        This method crops an image to the table region defined by the bounding
        box and sends it to OpenAI's vision API to extract the table content.

        Args:
            image_path (Union[str, Path]): Path to the image file
            bbox (Dict[str, int]): Table bounding box coordinates

        Returns:
            str: Extracted table in markdown format
        """
        # Crop the image to the table area
        with Image.open(image_path) as img:
            cropped = img.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
            cropped_path = self.table_images_dir / f"table_{Path(image_path).stem}.png"
            cropped.save(cropped_path)

        return self._call_vision_api(cropped_path, self.table_extraction_prompt)

    def process_pdf(self, pdf_path: Union[str, Path]) -> Tuple[str, List[Path]]:
        """Process a PDF file and convert it to markdown format with table extraction.

        This is the main method of the PDFProcessor class. It processes a PDF file
        by converting each page to an image, detecting tables using the vision API,
        extracting table content and converting to markdown, saving tables as CSV
        files, and processing regular text content.

        The method implements a comprehensive approach to PDF processing:
        1. Converts each page to a high-resolution image
        2. Detects tables using OpenAI's vision API
        3. Extracts table content and converts to markdown
        4. Saves tables as CSV files for further analysis
        5. Processes regular text content

        Args:
            pdf_path (Union[str, Path]): Path to the PDF file

        Returns:
            Tuple[str, List[Path]]: A tuple containing:
                - markdown_text (str): The complete markdown text of the PDF
                - table_files (List[Path]): List of paths to extracted table CSV files

        Raises:
            FileNotFoundError: If the PDF file does not exist
            Exception: For other processing errors

        Example:
            >>> processor = PDFProcessor()
            >>> markdown_text, table_files = processor.process_pdf("document.pdf")
            >>> print(markdown_text)  # The converted markdown text
            >>> for table_file in table_files:
            ...     print(f"Table saved to: {table_file}")
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

        markdown_text = []
        table_files = []

        try:
            for page_num in range(len(doc)):
                # Use 1-based page numbering for user-facing messages
                user_page_num = page_num + 1
                print(f"Processing page {user_page_num}/{len(doc)}...")
                page = doc[page_num]
                
                # Save page as image
                image_path = self._save_page_image(page, page_num)
                
                # Add delay between API calls
                time.sleep(self.api_delay)
                
                # Detect tables using vision API
                # Pass the 1-based page number to the _detect_table method
                table_bbox = self._detect_table(image_path, page_num=user_page_num)
                if table_bbox:
                    print(f"\nTable detected on page {user_page_num}!")
                    print(f"Bounding box coordinates:")
                    print(f"  Top-left: ({table_bbox['x1']}, {table_bbox['y1']})")
                    print(f"  Bottom-right: ({table_bbox['x2']}, {table_bbox['y2']})")
                    print(f"  Width: {table_bbox['x2'] - table_bbox['x1']}px")
                    print(f"  Height: {table_bbox['y2'] - table_bbox['y1']}px")
                
                if table_bbox:
                    try:
                        # Add delay between API calls
                        time.sleep(self.api_delay)
                        
                        # Extract and process table
                        table_text = self._extract_table(image_path, table_bbox)
                        markdown_table = self._convert_table_to_markdown(table_text)
                        csv_path = self._save_table_as_csv(markdown_table, page_num, 1)
                        if csv_path:
                            table_files.append(csv_path)
                        markdown_text.append(markdown_table)
                    except Exception as e:
                        print(f"Error processing table on page {user_page_num}: {str(e)}")
                        # Fall back to regular text processing
                        text = page.get_text()
                        markdown_text.append(text)
                else:
                    # Process regular page content
                    text = page.get_text()
                    markdown_text.append(text)

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        finally:
            doc.close()

        return '\n\n'.join(markdown_text), table_files

if __name__ == "__main__":
    processor = PDFProcessor()
    pdf_path = "data/sample_file.pdf"
    processor.process_pdf(pdf_path) 