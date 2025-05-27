from fastapi import APIRouter, HTTPException, Depends, Request
from app.services.prediction import PredictionService
from app.models import PredictionRequest, PredictionResponse, PredictionItem

router = APIRouter()


def get_prediction_service(request: Request) -> PredictionService:
    """Dependency to get the prediction service with loaded models"""
    models = request.app.state.model_loader.get_models()
    return PredictionService(models)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
        request: PredictionRequest,
        prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Generate predictions based on input parameters

    Returns a list of predictions with date and predicted value for each day
    """
    try:
        # Convertir news_content de Pydantic models a dicts
        news_content_dict = [
            {"content": item.content, "date": item.date}
            for item in request.news_content
        ]

        predictions = await prediction_service.generate_predictions(
            news_content=news_content_dict,
            prediction_horizon=request.prediction_horizon
        )

        return PredictionResponse(
            status="success",
            message=f"Generated {len(predictions)} predictions for {request.prediction_horizon} days",
            predictions=[PredictionItem(date=pred['date'], value=pred['value']) for pred in predictions])

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")