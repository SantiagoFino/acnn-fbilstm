import pickle
import torch
import joblib
from pathlib import Path

from app.core.settings import settings
from transformers import pipeline


class ModelLoader:
    def __init__(self):
        self.arima_model = None
        self.pca_model = None
        self.transformer_model = None
        self.target_scaler = None
        self.text_scaler = None
        self.ts_scaler = None
        self.news_embedder = None
        self.news_summarizer = None

    async def load_all_models(self):
        """Load all required models"""
        try:
            await self.load_arima_model()
            await self.load_pca_model()
            await self.load_acnn_fBiLSTM_model()
            await self.load_ts_scaler()
            await self.load_target_scaler()
            await self.load_text_scaler()
            await self.load_news_embedder()
            await self.load_news_summarizer()
            print("All models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    async def load_arima_model(self):
        """Load ARIMA"""
        model_path = Path(settings.ARIMA_MODEL_PATH)
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.arima_model = pickle.load(f)
            print("ARIMA model loaded")
        else:
            print(f"ARIMA model not found at {model_path}")

    async def load_pca_model(self):
        """Load PCA model"""
        model_path = Path(settings.PCA_MODEL_PATH)
        if model_path.exists():
            self.pca_model = joblib.load(model_path)
            print("PCA model loaded")
        else:
            print(f"PCA model not found at {model_path}")

    async def load_acnn_fBiLSTM_model(self):
        """Load PyTorch transformer model"""
        model_path = Path(settings.ACNN_fBiLSTM_PATH)
        if model_path.exists():
            self.transformer_model = torch.load(model_path, map_location='cpu')
            self.transformer_model.eval()
            print("Transformer model loaded")
        else:
            print(f"Transformer model not found at {model_path}")

    async def load_target_scaler(self):
        """Load data scaler"""
        scaler_path = Path(settings.TARGET_SCALER_PATH)
        if scaler_path.exists():
            self.target_scaler = joblib.load(scaler_path)
            print("Scaler loaded")
        else:
            print(f"Scaler not found at {scaler_path}")

    async def load_text_scaler(self):
        """Load text scaler"""
        scaler_path = Path(settings.TEXT_SCALER_PATH)
        if scaler_path.exists():
            self.text_scaler = joblib.load(scaler_path)
            print("Text scaler loaded")
        else:
            print(f"Text scaler not found at {scaler_path}")

    async def load_ts_scaler(self):
        """Load time series scaler"""
        scaler_path = Path(settings.TS_SCALER_PATH)
        if scaler_path.exists():
            self.ts_scaler = joblib.load(scaler_path)
            print("Time series scaler loaded")
        else:
            print(f"Time series scaler not found at {scaler_path}")

    async def load_news_embedder(self):
        """Load news text embedding model"""
        try:
            self.news_embedder = pipeline(
                'feature-extraction',
                model=settings.NEWS_EMBEDDING_MODEL,
                tokenizer=settings.NEWS_EMBEDDING_MODEL
            )
            print("News embedding model loaded")
        except Exception as e:
            print(f"Failed to load news embedder: {e}")

    async def load_news_summarizer(self):
        """Load news text summarizer"""
        try:
            self.news_summarizer = pipeline(
                'feature-extraction',
                model=settings.NEWS_SUMMARIZER_MODEL,
                tokenizer=settings.NEWS_SUMMARIZER_MODEL,
                framework='pt',
                device=-1,
                clean_up_tokenization_spaces=True
            )
            print("News summarizer model loaded")
        except Exception as e:
            print(f"Failed to load news summarizer: {e}")

    def get_models(self):
        """Return all loaded models"""
        return {
            'arima': self.arima_model,
            'pca_model': self.pca_model,
            'ACNN-fBiLSTM': self.transformer_model,
            'target_scaler': self.target_scaler,
            'text_scaler': self.text_scaler,
            'ts_scaler': self.ts_scaler,
            'news_embedder': self.news_embedder,
            'news_summarizer': self.news_summarizer
        }
