from fastapi import APIRouter
from ..controllers import roadmap_controller
from ..models.models import generateRoadMapRequest

router = APIRouter()


@router.post("/generateRoadmapRequest")
def generate_roadmap_request(payload: generateRoadMapRequest):
    return roadmap_controller.generate_roadmap(payload)
