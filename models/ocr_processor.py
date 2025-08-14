# models/ocr_processor.py - PyMuPDF + Qwen2-VL Implementation (RAM Optimized)
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
import os
import gc
from typing import Union, Dict
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    RAM-optimized OCR utility using PyMuPDF + Qwen2-VL
    Handles both PDF text extraction and vision-based OCR with minimal memory footprint
    """
    
    def __init__(self):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_loaded = False
        self._lazy_load_enabled = True  # Enable lazy loading by default
        
    def _get_device(self):
        """Determine the best available device with memory optimization"""
        if torch.cuda.is_available():
            # Check available GPU memory
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if gpu_memory < 8:  # Less than 8GB GPU memory
                    logger.info(f"GPU memory limited ({gpu_memory:.1f}GB), using CPU for better memory management")
                    return "cpu"
                return "cuda"
            except:
                return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _clear_memory(self):
        """Clear GPU/CPU memory and garbage collect"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def load_model(self, force_reload=False):
        """Load Qwen2.5-VL-3B-Instruct model with memory optimization"""
        if self.model_loaded and not force_reload:
            return True
            
        try:
            logger.info("Loading Qwen2-VL model for OCR from local path...")

            # Resolve local model directory: models/qwen_vlm
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, 'qwen_vlm')

            if not os.path.isdir(model_dir):
                logger.error(f"Local Qwen VLM folder not found at: {model_dir}")
                self.model_loaded = False
                return False

            # Memory optimization: Use lower precision and device mapping
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load model with memory optimization
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True,
                low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                offload_folder="temp_offload" if self.device == "cpu" else None  # Offload to disk if CPU
            )

            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

            if self.device != "cuda":
                self.model.to(self.device)

            self.model_loaded = True
            logger.info("Qwen2-VL local model loaded successfully with memory optimization")
            return True

        except Exception as e:
            logger.error(f"Failed to load local Qwen2-VL model: {str(e)}")
            self.model_loaded = False
            return False
    
    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on PIL Image object using Qwen2-VL with memory optimization"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize large images to reduce memory usage
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {image.size} to {new_size} for memory optimization")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Bu görüntüdeki tüm metni oku ve Türkçe olarak çıkar. Metni olduğu gibi, herhangi bir açıklama olmadan ver."}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            
            # Memory optimization: Use smaller max_new_tokens and enable memory efficient attention
            with torch.no_grad():  # Disable gradient computation
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,  # Reduced from 1024
                    do_sample=False,  # Deterministic generation
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            cleaned_response = self._extract_assistant_content(response[0])
            
            # Clear inputs from memory
            del inputs, generated_ids
            self._clear_memory()
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Qwen2-VL OCR failed: {str(e)}", exc_info=True)
            return f"OCR Error: {str(e)}"

    def _extract_assistant_content(self, response: str) -> str:
        """Extract only the assistant's generated content and hide system/user prompts."""
        if not response:
            return ""
        text = response.strip()
        lower = text.lower()
        # Prefer content after the last 'assistant' marker if present
        if 'assistant' in lower:
            idx = lower.rfind('assistant')
            tail = text[idx:]
            lines = tail.splitlines()
            if len(lines) > 1:
                content = "\n".join(lines[1:]).strip()
                if content:
                    return content
        # Fallback: remove any leading role headers or generic system descriptions
        skip_headers = {"system", "user", "assistant"}
        filtered_lines = []
        for line in text.splitlines():
            l = line.strip()
            if l.lower() in skip_headers:
                continue
            if l.startswith("You are a helpful assistant"):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Union[str, bool, int]]:
        try:
            if not os.path.exists(pdf_path):
                return {'success': False, 'error': 'PDF file not found', 'text': '', 'method': 'none', 'page_count': 0}
            
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            extracted_text = ""
            method_used = "text_extraction"
            
            # Try text extraction first
            for page in doc:
                extracted_text += page.get_text() + "\n"
            
            # If text extraction yields little content, use OCR
            if len(extracted_text.strip()) < 100:
                logger.info(f"PDF appears to be scanned, using Qwen2-VL OCR for {pdf_path}")
                if not self.model_loaded:
                    if not self.load_model():
                        doc.close()
                        return {'success': False, 'error': 'Qwen2-VL model not available', 'text': '', 'method': 'none', 'page_count': page_count}
                
                # Process pages in batches to manage memory
                extracted_text = self._ocr_pdf_pages_optimized(doc)
                method_used = "qwen2_vl_ocr"
            
            doc.close()
            return {'success': True, 'text': extracted_text.strip(), 'method': method_used, 'page_count': page_count, 'error': None}
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {'success': False, 'error': str(e), 'text': '', 'method': 'none', 'page_count': 0}
    
    def _ocr_pdf_pages_optimized(self, doc) -> str:
        """OCR PDF pages with memory optimization"""
        all_text = ""
        batch_size = 3  # Process 3 pages at a time to manage memory
        
        for i in range(0, len(doc), batch_size):
            batch_end = min(i + batch_size, len(doc))
            batch_text = ""
            
            for page_num in range(i, batch_end):
                page = doc[page_num]
                try:
                    # Optimize image resolution for memory
                    mat = fitz.Matrix(1.5, 1.5)  # Reduced from 2x2 to 1.5x1.5
                    pix = page.get_pixmap(matrix=mat)
                    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    
                    page_text = self._perform_ocr(image)
                    batch_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    # Clear page-specific memory
                    del pix, image
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    batch_text += f"\n--- Page {page_num + 1} ---\nError: {str(e)}\n"
            
            all_text += batch_text
            
            # Clear batch memory and force garbage collection
            del batch_text
            self._clear_memory()
            
            # Small delay to allow memory cleanup
            import time
            time.sleep(0.1)
        
        return all_text
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Union[str, bool]]:
        try:
            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Image file not found', 'text': ''}
            
            _, ext = os.path.splitext(image_path.lower())
            if ext not in self.supported_image_formats:
                return {'success': False, 'error': f'Unsupported image format: {ext}', 'text': ''}
            
            if not self.model_loaded:
                if not self.load_model():
                    return {'success': False, 'error': 'Qwen2-VL model not available', 'text': ''}
            
            image = Image.open(image_path).convert("RGB")
            extracted_text = self._perform_ocr(image)
            
            # Clear image from memory
            del image
            self._clear_memory()
            
            return {'success': True, 'text': extracted_text, 'error': None}
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {'success': False, 'error': str(e), 'text': ''}

    def get_text_from_file(self, file_path: str) -> Dict[str, Union[str, bool, int]]:
        if not os.path.exists(file_path):
            return {'success': False, 'error': 'File not found', 'text': ''}
        
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in self.supported_image_formats:
            result = self.extract_text_from_image(file_path)
            if result.get('success'):
                result.update({'method': 'qwen2_vl_ocr', 'page_count': 1})
            return result
        else:
            return {'success': False, 'error': f'Unsupported file format: {ext}', 'text': ''}
    
    def is_model_ready(self) -> bool:
        return self.model_loaded
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model_loaded:
            del self.model
            del self.processor
            del self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.model_loaded = False
            self._clear_memory()
            logger.info("Model unloaded to free memory")
    
    def get_memory_usage(self) -> Dict[str, str]:
        """Get current memory usage information"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            memory_info['gpu_reserved'] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            memory_info['gpu_free'] = f"{torch.cuda.memory_reserved() / 1024**3 - torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        
        import psutil
        process = psutil.Process()
        memory_info['ram_used'] = f"{process.memory_info().rss / 1024**3:.2f} GB"
        memory_info['ram_percent'] = f"{process.memory_percent():.1f}%"
        
        return memory_info

# Global instance with lazy loading
ocr_processor = OCRProcessor()