from datetime import datetime, timedelta
from typing import Any
from app.core.settings import settings
from app.ml_models.pipeline import MainPipeline


class PredictionService:
    def __init__(self, models: dict[str, Any]):
        self.starting_date = datetime.strptime(settings.STARTING_DATE, '%Y-%m-%d')
        self.models = models

    async def generate_predictions(self, news_content: list[dict[str, Any]], prediction_horizon: int
                                   ) -> list[dict[str, float]]:
        """
        Generate predictions for the specified horizon
        """
        try:
            model_horizon = await self._prepare_horizon(prediction_horizon)
            news_data = await self._prepare_news_data(model_horizon, news_content)

            predictions = await self.predict(news_data=news_data)
            return predictions

        except Exception as e:
            print(f"Error generating predictions: {e}")
            raise

    async def _prepare_horizon(self, prediction_horizon: int) -> int:
        """
        Computes the true prediction horizon based on the starting date and the prediction horizon
        """
        days_difference = int((datetime.today() - self.starting_date).days)
        model_horizon = days_difference + prediction_horizon
        return int(model_horizon)

    async def _prepare_news_data(self, model_horizon: int, news_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Fill the missing dates in the prediction horizon with empty news content
        """
        dates = [self.starting_date + timedelta(days=i) for i in range(model_horizon)]

        existing_entries = {}
        for entry in news_data:
            date_obj = datetime.strptime(entry['date'], '%Y-%m-%d')
            existing_entries[date_obj.date()] = entry['content']

        result = []
        for dt in sorted(dates):
            date_key = dt.date()
            if date_key in existing_entries:
                result.append({
                    'content': existing_entries[date_key],
                    'date': dt
                })
            else:
                result.append({
                    'content': "",
                    'date': dt
                })
        return result

    async def predict(self, news_data: list[dict[str, Any]], horizon):
        """
        Generate predictions based on the news data
        """
        pipeline = MainPipeline(self.models)
        try:
            predictions = pipeline.process_ecopetrol_data(news_data, horizon)
        except Exception as e:
            print(f"Error making the prediction: {e}")
            raise
        return predictions
