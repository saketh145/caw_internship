"""
Multi-format document ingestion and processing
Supports PDF, DOCX, TXT, HTML, and Markdown files
"""
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Document processing imports
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import markdown

from ..config import settings
from loguru import logger

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    source_path: str
    file_type: str
    file_size: int
    created_at: datetime
    modified_at: datetime
    document_hash: str
    total_pages: Optional[int] = None
    total_words: Optional[int] = None

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None

class DocumentProcessor:
    """Main document processing class with multi-format support"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt', '.html', '.htm', '.md', '.markdown'}
        
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and return chunks with metadata
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing chunks and metadata
        """
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Extract text based on file type
            text_content = await self._extract_text(file_path, file_extension)
            
            # Generate metadata
            metadata = self._generate_metadata(file_path, file_extension, text_content)
            
            # Create intelligent chunks
            chunks = self._create_intelligent_chunks(text_content, metadata)
            
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            
            return {
                "metadata": metadata,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    async def _extract_text(self, file_path: str, file_extension: str) -> str:
        """Extract text content based on file type"""
        
        if file_extension == '.pdf':
            return await self._extract_pdf_text(file_path)
        elif file_extension == '.docx':
            return await self._extract_docx_text(file_path)
        elif file_extension in ['.html', '.htm']:
            return await self._extract_html_text(file_path)
        elif file_extension in ['.md', '.markdown']:
            return await self._extract_markdown_text(file_path)
        elif file_extension == '.txt':
            return await self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Add page markers for better chunking
                    text_content += f"\n\n[PAGE {page_num + 1}]\n{page_text}\n"
            return text_content.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise
    
    async def _extract_html_text(self, file_path: str) -> str:
        """Extract text from HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            logger.error(f"Error extracting HTML text: {str(e)}")
            raise
    
    async def _extract_markdown_text(self, file_path: str) -> str:
        """Extract text from Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to HTML then extract text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Error extracting Markdown text: {str(e)}")
            raise
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting TXT text: {str(e)}")
            raise
    
    def _generate_metadata(self, file_path: str, file_extension: str, content: str) -> DocumentMetadata:
        """Generate comprehensive metadata for the document"""
        file_stat = os.stat(file_path)
        
        # Calculate document hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Count words
        word_count = len(content.split())
        
        # Extract page count for PDFs
        page_count = None
        if file_extension == '.pdf':
            page_count = content.count('[PAGE')
        
        return DocumentMetadata(
            source_path=file_path,
            file_type=file_extension,
            file_size=file_stat.st_size,
            created_at=datetime.fromtimestamp(file_stat.st_ctime),
            modified_at=datetime.fromtimestamp(file_stat.st_mtime),
            document_hash=content_hash,
            total_pages=page_count,
            total_words=word_count
        )
    
    def _create_intelligent_chunks(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """
        Create intelligent chunks using multiple strategies:
        1. Semantic chunking based on paragraph boundaries
        2. Sliding window with overlap for context preservation
        3. Dynamic chunk sizing based on content type
        """
        chunks = []
        
        # Strategy 1: Paragraph-based chunking for better semantic coherence
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= settings.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Create chunk if we have content
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, settings.chunk_overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                current_start = len(text) - len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split on double newlines, but preserve other paragraph indicators
        paragraphs = []
        
        # Split by various paragraph indicators
        potential_splits = text.split('\n\n')
        
        for split in potential_splits:
            split = split.strip()
            if split:
                # Further split very long paragraphs
                if len(split) > settings.max_chunk_size * 0.8:
                    # Split on sentence boundaries
                    sentences = split.split('. ')
                    current_para = ""
                    
                    for sentence in sentences:
                        if len(current_para + sentence) < settings.max_chunk_size * 0.8:
                            current_para += sentence + ". "
                        else:
                            if current_para:
                                paragraphs.append(current_para.strip())
                            current_para = sentence + ". "
                    
                    if current_para:
                        paragraphs.append(current_para.strip())
                else:
                    paragraphs.append(split)
        
        return paragraphs
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good breaking point (sentence or word boundary)
        overlap_text = text[-overlap_size:]
        
        # Find the start of the last complete sentence
        last_period = overlap_text.rfind('. ')
        if last_period > overlap_size * 0.5:
            return overlap_text[last_period + 2:]
        
        # Fall back to word boundary
        last_space = overlap_text.rfind(' ')
        if last_space > overlap_size * 0.5:
            return overlap_text[last_space + 1:]
        
        return overlap_text
    
    def _create_chunk(self, content: str, chunk_index: int, start_char: int, metadata: DocumentMetadata) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata"""
        chunk_id = f"{metadata.document_hash}_{chunk_index}"
        
        # Extract page number if available (for PDFs)
        page_number = None
        if '[PAGE' in content:
            import re
            page_match = re.search(r'\[PAGE (\d+)\]', content)
            if page_match:
                page_number = int(page_match.group(1))
        
        chunk_metadata = {
            "source_file": metadata.source_path,
            "file_type": metadata.file_type,
            "chunk_index": chunk_index,
            "document_hash": metadata.document_hash,
            "created_at": datetime.now().isoformat(),
            "word_count": len(content.split()),
            "char_count": len(content)
        }
        
        if page_number:
            chunk_metadata["page_number"] = page_number
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=chunk_metadata,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=start_char + len(content),
            page_number=page_number
        )

# Export the main processor
document_processor = DocumentProcessor()
