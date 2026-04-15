from typing import Dict
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import field_serializer
from pydantic import field_validator
from scenedetect import FrameTimecode


class Scene(BaseModel):
    start: FrameTimecode
    end: FrameTimecode
    video_id: Optional[str] = None
    scene_idx: Optional[int] = -1
    audio_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @field_serializer("start", "end")
    def serialize_frametimecode(
        self, value: FrameTimecode
    ) -> Dict[str, Union[str, float]]:
        """Serialize FrameTimecode as a string."""
        ret = {"timecode": value.get_timecode(), "fps": value.get_framerate()}
        return ret

    @field_validator("start", "end", mode="before")
    @classmethod
    def deserialize_frametimecode(
        cls, value: Union[FrameTimecode, Dict[str, Union[str, float]]]
    ) -> FrameTimecode:
        """Deserialize FrameTimecode from a string."""
        if isinstance(value, Dict):
            timecode = value["timecode"]
            fps = value["fps"]
            return FrameTimecode(timecode=timecode, fps=fps)
        else:
            return value