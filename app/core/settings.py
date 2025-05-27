from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Prediction with ACNN-fBiLSTM"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    MODEL_DIR: str = "app/ml_models/trained_models"
    ARIMA_MODEL_PATH: str = f"{MODEL_DIR}/arima_model.pkl"
    PCA_MODEL_PATH: str = f"{MODEL_DIR}/pca_model.pkl"
    ACNN_fBiLSTM_PATH: str = f"{MODEL_DIR}/acnn-fBiLSTM.pkl"
    TARGET_SCALER_PATH: str = f"{MODEL_DIR}/target_scaler.pkl"
    TEXT_SCALER_PATH: str = f"{MODEL_DIR}/text_scaler.pkl"
    TS_SCALER_PATH: str = f"{MODEL_DIR}/ts_scaler.pkl"

    MAX_PREDICTION_HORIZON: int = 365
    DEFAULT_PREDICTION_HORIZON: int = 30
    STARTING_DATE: str = "2025-05-14"

    NEWS_EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    NEWS_SUMMARIZER_MODEL: str = "ELiRF/mt5-base-dacsa-es"

    class Config:
        env_file = ".env"


settings = Settings()
