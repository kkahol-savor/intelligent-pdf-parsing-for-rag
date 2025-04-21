import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import fitz
import sys
import os
import json
import base64
from PIL import Image
import io

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_to_markdown import PDFProcessor

@pytest.fixture
def mock_env_vars():
    with patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint',
        'AZURE_OPENAI_KEY': 'test-key',
        'AZURE_OPENAI_DEPLOYMENT': 'test-deployment',
        'IMAGE_DESCRIPTION_PROMPT': 'test prompt'
    }):
        yield

@pytest.fixture
def mock_fitz():
    with patch('fitz.open') as mock_open:
        # Create a mock document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        
        # Mock page methods
        mock_page.get_images.return_value = [(0, "jpeg", 0, 0, 0, 0, 0, 0, 0, 0, 0)]
        mock_page.get_image_rects.return_value = [fitz.Rect(0, 0, 100, 100)]
        mock_page.get_text.return_value = "Test page content"
        
        # Mock get_pixmap to create and save a real image
        def mock_get_pixmap(*args, **kwargs):
            mock_pixmap = MagicMock()
            def mock_save(path):
                # Create a small test image and save it
                img = Image.new('RGB', (100, 100), color='red')
                img.save(path)
            mock_pixmap.save = mock_save
            mock_pixmap.tobytes = MagicMock(return_value=b"fake image data")
            return mock_pixmap
        
        mock_page.get_pixmap = mock_get_pixmap
        
        # Mock document methods
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.extract_image.return_value = {"image": b"fake image data"}
        
        mock_open.return_value = mock_doc
        yield mock_open

@pytest.fixture
def mock_requests():
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test image description'}}]
        }
        mock_post.return_value = mock_response
        yield mock_post

@pytest.fixture
def processor(mock_env_vars, tmp_path):
    # Create a processor with temporary directories
    processor = PDFProcessor()
    processor.output_dir = tmp_path / "output_images"
    processor.output_dir.mkdir(exist_ok=True)
    processor.tables_dir = tmp_path / "tables"
    processor.tables_dir.mkdir(exist_ok=True)
    processor.table_images_dir = processor.tables_dir / "images"
    processor.table_images_dir.mkdir(exist_ok=True)
    return processor

def test_init(processor):
    assert processor.endpoint == 'https://test-endpoint'
    assert processor.api_key == 'test-key'
    assert processor.deployment == 'test-deployment'
    assert processor.image_prompt == 'test prompt'

def test_encode_image(processor, tmp_path):
    # Create a valid test image
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    
    encoded = processor._encode_image(str(img_path))
    assert isinstance(encoded, str)
    assert len(encoded) > 0

def test_detect_table(processor, mock_requests, tmp_path):
    # Create a valid test image
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    
    # Mock the vision API response for table detection
    mock_requests.return_value.json.return_value = {
        'choices': [{'message': {'content': json.dumps({
            'x1': 100,
            'y1': 100,
            'x2': 200,
            'y2': 200
        })}}]
    }
    
    bbox = processor._detect_table(img_path)
    assert bbox is not None
    assert bbox['x1'] == 100
    assert bbox['y1'] == 100
    assert bbox['x2'] == 200
    assert bbox['y2'] == 200

def test_extract_table(processor, mock_requests, tmp_path):
    # Create a valid test image
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    
    # Mock the vision API response for table extraction
    mock_requests.return_value.json.return_value = {
        'choices': [{'message': {'content': '| Header1 | Header2 |\n|---------|----------|\n| Data1   | Data2   |'}}]
    }
    
    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    table_text = processor._extract_table(img_path, bbox)
    
    assert table_text is not None
    assert '| Header1 | Header2 |' in table_text
    assert '| Data1   | Data2   |' in table_text

def test_convert_table_to_markdown(processor):
    table_text = '| Header1 | Header2 |\n|---------|----------|\n| Data1   | Data2   |'
    markdown = processor._convert_table_to_markdown(table_text)
    
    assert markdown == table_text

def test_save_table_as_csv(processor, tmp_path):
    table_text = '| Header1 | Header2 |\n|---------|----------|\n| Data1   | Data2   |'
    processor.tables_dir = tmp_path / "tables"
    processor.tables_dir.mkdir(exist_ok=True)
    csv_path = processor._save_table_as_csv(table_text, 0, 1)
    
    assert csv_path is not None
    assert csv_path.exists()
    assert csv_path.suffix == '.csv'
    
    # Verify CSV content
    content = csv_path.read_text()
    assert 'Header1,Header2' in content
    assert 'Data1,Data2' in content

def test_process_pdf(processor, mock_fitz, mock_requests, tmp_path):
    # Create necessary directories
    output_dir = tmp_path / "output_images"
    tables_dir = tmp_path / "tables"
    table_images_dir = tables_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Update processor directories
    processor.output_dir = output_dir
    processor.tables_dir = tables_dir
    processor.table_images_dir = table_images_dir
    
    # Create a temporary PDF file
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%EOF")  # Minimal valid PDF content
    
    # Mock the vision API responses for both table detection and extraction
    def mock_response(*args, **kwargs):
        mock_resp = Mock()
        mock_resp.status_code = 200
        
        # Get the request data
        request_data = kwargs.get('json', {})
        messages = request_data.get('messages', [])
        
        # Check if this is a table detection or extraction request
        is_table_detection = any('table_detection' in msg.get('content', '') for msg in messages)
        
        if is_table_detection:
            # Mock response for table detection - return just the bbox coordinates
            mock_resp.json.return_value = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'x1': 100,
                            'y1': 100,
                            'x2': 200,
                            'y2': 200
                        })
                    }
                }]
            }
        else:
            # Mock response for table extraction
            mock_resp.json.return_value = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'markdown': '| Header1 | Header2 |\n|---------|----------|\n| Data1   | Data2   |',
                            'csv': 'Header1,Header2\nData1,Data2'
                        })
                    }
                }]
            }
        return mock_resp
    
    mock_requests.side_effect = mock_response
    
    # Process the PDF
    markdown_text, table_files = processor.process_pdf(pdf_path)
    
    # Check if output files were created
    page_image = output_dir / "page_0.png"
    assert page_image.exists(), f"Page image not found at {page_image}"
    
    # Check if markdown text was generated
    assert markdown_text, "No markdown text was generated"
    
    # If tables were detected and processed
    if table_files:
        assert len(table_files) > 0, "No table files were created"
        
        # Verify table file content
        table_file = table_files[0]
        assert table_file.exists(), f"Table file not found at {table_file}"
        assert table_file.suffix == '.csv', "Table file is not a CSV file"
        
        # Verify CSV content
        content = table_file.read_text()
        assert 'Header1,Header2' in content, "CSV header not found"
        assert 'Data1,Data2' in content, "CSV data not found"

def test_process_pdf_error_handling(processor, mock_fitz, mock_requests, tmp_path):
    # Test with non-existent PDF file
    with pytest.raises(FileNotFoundError):
        processor.process_pdf(tmp_path / "nonexistent.pdf")
    
    # Test with invalid PDF file
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("Not a PDF file")
    
    # Mock fitz.open to raise an error for invalid PDF
    mock_fitz.side_effect = Exception("Invalid PDF file")
    
    with pytest.raises(Exception) as exc_info:
        processor.process_pdf(invalid_pdf)
    assert "Error processing PDF" in str(exc_info.value)

def test_api_error_handling(processor, mock_requests, tmp_path):
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    
    # Test rate limit handling
    mock_requests.return_value.status_code = 429
    mock_requests.return_value.headers = {'Retry-After': '1'}
    
    with pytest.raises(Exception) as exc_info:
        processor._detect_table(img_path)
    assert "Failed to call Vision API" in str(exc_info.value)
    
    # Test other API errors
    mock_requests.return_value.status_code = 500
    with pytest.raises(Exception) as exc_info:
        processor._detect_table(img_path)
    assert "Failed to call Vision API" in str(exc_info.value) 