# models/model_manager.py - Updated version
import torch
from transformers import pipeline
from llama_cpp import Llama
import os
import logging
from .ocr_processor import ocr_processor
from .ner_processor import ner_processor
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.classifier = None
            self.summarizer = None
            self.ocr_processor = ocr_processor
            self.ner_processor = ner_processor
            self.device = self._get_device()
            self.initialized = True
        
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_classifier(self):
        """Load XLM-RoBERTa classifier for CPU/GPU"""
        if self.classifier is not None:
            logger.info("Classifier already loaded, skipping...")
            return True
            
        try:
            logger.info("Loading classifier model...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="joeddav/xlm-roberta-large-xnli",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Classifier loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False
    
    def load_summarizer(self, model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"):
        """Load LLaMA GGUF model from a local path"""
        if self.summarizer is not None:
            logger.info("Summarizer already loaded, skipping...")
            return True
            
        try:
            logger.info("Loading summarizer model...")
            
            if not os.path.exists(model_path):
                logger.error(f"Summarizer model not found at {model_path}")
                logger.error("Please make sure you have downloaded the model and placed it in the 'models' directory.")
                return False

            # Configure based on available resources
            if self.device == "cuda":
                gpu_layers = 35  # Adjust based on VRAM
            else:
                gpu_layers = 35  # Adjust based on VRAM
                
            self.summarizer = Llama(
                model_path=model_path,
                n_ctx=8192,  # Context length - increased for better capacity
                n_gpu_layers=gpu_layers,
                verbose=False # Set back to False
            )
            logger.info("Summarizer loaded successfully")
            if self.summarizer is None:
                logger.error("Summarizer object is None after loading.")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to load summarizer: {e}", exc_info=True)
            return False
    
    def load_ner_model(self):
        """Load NER model"""
        try:
            logger.info("Loading NER model...")
            success = self.ner_processor.load_model()
            if success:
                logger.info("NER model loaded successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            return False
    
    def load_ocr_model(self):
        """Load OCR model with memory optimization"""
        try:
            logger.info("Loading OCR model with memory optimization...")
            success = self.ocr_processor.load_model()
            if success:
                logger.info("OCR model loaded successfully with memory optimization")
                # Clear memory after loading
                self.ocr_processor._clear_memory()
            return success
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            return False

    def load_all_models(self):
        """Load all models - only once per application lifecycle"""
        if ModelManager._models_loaded:
            logger.info("Models already loaded, skipping initialization...")
            return True
            
        logger.info(f"Using device: {self.device}")
        
        classifier_ok = self.load_classifier()
        summarizer_ok = self.load_summarizer()
        ner_ok = self.load_ner_model()
        ocr_ok = self.load_ocr_model()
        
        logger.info(f"Model loading status:")
        logger.info(f"  Classifier: {'✓' if classifier_ok else '✗'}")
        logger.info(f"  Summarizer: {'✓' if summarizer_ok else '✗'}")
        logger.info(f"  NER: {'✓' if ner_ok else '✗'}")
        logger.info(f"  OCR: {'✓' if ocr_ok else '✗'}")
        
        # Mark models as loaded if all successful
        if classifier_ok and summarizer_ok and ner_ok and ocr_ok:
            ModelManager._models_loaded = True
            logger.info("All models loaded successfully and cached!")
        
        return classifier_ok and summarizer_ok and ner_ok and ocr_ok

    def cleanup_models(self):
        """Free cached GPU/accelerator memory where possible."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear OCR model memory if loaded
            if hasattr(self.ocr_processor, 'model_loaded') and self.ocr_processor.model_loaded:
                self.ocr_processor._clear_memory()
                
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception:
            pass
    
    def ensure_classifier_loaded(self):
        """Ensure classifier is loaded before use"""
        if self.classifier is None:
            logger.info("Lazy loading classifier model...")
            return self.load_classifier()
        return True
    
    def ensure_summarizer_loaded(self):
        """Ensure summarizer is loaded before use"""
        if self.summarizer is None:
            logger.info("Lazy loading summarizer model...")
            return self.load_summarizer()
        return True
    
    def get_text_from_file(self, file_path):
        """Convenience method to extract text from any supported file"""
        return self.ocr_processor.get_text_from_file(file_path)
    
    def analyze_entities(self, text):
        """Convenience method for NER analysis"""
        return self.ner_processor.analyze_entities(text)