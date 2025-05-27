from typing import Any
import numpy as np
import pandas as pd


class NewsTransformationPipeline:
    MAX_LENGTH = 512

    def __init__(self, models: dict[str, Any]):
        self.embedder = models['news_embedder']
        self.summarizer = models['news_summarizer']
        self.pca_model = models['pca_model']

    def summarize_news(self, news_text: str):
        """Summarize the text of a new"""
        inputs = self.summarizer.tokenizer.tokenize(news_text)
        if len(inputs) <= self.MAX_LENGTH:
            print(f"Text is already less than {self.MAX_LENGTH} characters, skipping summarization")
            return news_text
        try:
            summary = self.summarizer(
                news_text,
                min_length=0,
                do_sample=False,
                truncation=True,
                max_length=572
            )
        except Exception as e:
            print(f"Error during summarization: {e}")
            return news_text
        return summary[0]['summary_text']

    def embed_news(self, news_text: str) -> np.ndarray:
        """Embed news texts using SentenceTransformer"""
        try:
            embeddings = self.embedder.encoder.encode(news_text)
        except Exception as e:
            print(f"Error during embedding: {e}")
            return np.zeros((1, 768))  # Return a zero vector if embedding fails
        return embeddings

    def apply_pca(self, embedding, n_components: int = 50) -> np.ndarray:
        """
        Apply PCA to reduce the dimensionality of embeddings
        """
        try:
            pca_embeddings = self.pca_model['pca_model'].transform(embedding)
        except Exception as e:
            print(f"Error during PCA transformation: {e}")
            return embedding[:n_components]
        return pca_embeddings

    def transform(self, news_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Transform news text into summary and embeddings
        """
        for new in news_data:
            summarized_text = self.summarize_news(new['content'])
            embedded_text = self.embed_news(summarized_text)
            reduction = self.apply_pca(embedded_text)
            new['content'] = reduction
        return news_data


class EcopetrolDataPipeline:
    def __init__(self, models: dict[str, Any]):
        self.arima_model = models['arima_model']

    def transform(self, steps: int = 1, alpha=0.05) -> pd.DataFrame:
        """
        Process news text to generate summary and embeddings
        """
        fitted_model = self.arima_model['fitted_model']
        forecast = fitted_model.forecast(steps=steps, alpha=alpha)
        df = pd.DataFrame(forecast, columns=['arima_processed', 'residuals', 'diff_1'])
        df['arima_processed'] = forecast
        df['residuals'] = [0 for _ in range(len(forecast))]
        df['diff_1'] = forecast.diff().fillna(0)
        return df


class MainPipeline:
    def __init__(self, models: dict[str, Any]):
        self.news_pipeline = NewsTransformationPipeline(models)
        self.ecopetrol_pipeline = EcopetrolDataPipeline(models)
        self.acnn_fBiLSTM_model = models.get('acnn_fBiLSTM_model')

    def process_news(self, news_texts: list[dict[str, Any]]) -> list[Any]:
        """Process news texts to get summaries and embeddings"""
        return self.news_pipeline.transform(news_texts)

    def process_ecopetrol_data(self, steps: int = 1, alpha=0.05) -> pd.DataFrame:
        """Process Ecopetrol data using ARIMA model"""
        return self.ecopetrol_pipeline.transform(steps, alpha)

    def process_acnn_fBiLSTM_data(self, news_data, horizon) -> pd.DataFrame:
        """
        Process data using ACNN-fBiLSTM model
        This method is a placeholder for future implementation
        """
        news = self.process_news(news_data)
        processed_eco_data = self.process_ecopetrol_data(horizon)
        result = self.acnn_fBiLSTM_model.predict(news, processed_eco_data)
        return result
