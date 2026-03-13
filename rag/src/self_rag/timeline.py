import time
import uuid
from typing import Dict, List, Optional, TypeAlias, Union

from pydantic import BaseModel, Field

ValueType: TypeAlias = Union[str, int, float, bool, uuid.UUID]
ArgumentType: TypeAlias = Union[ValueType, List[ValueType], Dict[str, ValueType], None]
MetricType: TypeAlias = Union[int, float]


class TimelineEvent(BaseModel):
    event: str
    tag: Optional[str] = ""
    start_time_ms: float
    end_time_ms: float

    def __lt__(self, other: "TimelineEvent") -> bool:
        return self.start_time_ms < other.start_time_ms

    def to_csv(self) -> str:
        return f"{self.event},{self.tag},{self.start_time_ms},{self.end_time_ms}"

    @staticmethod
    def get_csv_header() -> str:
        return "event,tag,start_time_ms,end_time_ms"


class Timeline(BaseModel):
    events: List[TimelineEvent] = []

    def add_event(self, timeline_event: TimelineEvent):
        self.events.append(timeline_event)

    def merge(self, other_timeline: "Timeline"):
        self.events.extend(other_timeline.events)

    def to_csv(self, tag: str = "") -> str:
        self.events.sort()
        lines = []
        for event in self.events:
            if tag != "":
                lines.append(f"{tag},{event.to_csv()}")
            else:
                lines.append(event.to_csv())
        return "\n".join(lines)

    @staticmethod
    def get_csv_header() -> str:
        return TimelineEvent.get_csv_header()


class JobDescriptor(BaseModel):
    job_id: uuid.UUID = Field(default_factory=uuid.uuid4)


class RequestContext(BaseModel):
    job_descriptor: JobDescriptor = Field(default_factory=JobDescriptor)
    timeline: Timeline = Timeline()

    def get_timeline_csv(self) -> str:
        return self.timeline.to_csv(tag=str(self.job_descriptor.job_id))

    @staticmethod
    def get_timeline_csv_header() -> str:
        return f"job_id,{Timeline.get_csv_header()}"


class TimelineTracer:
    def __init__(self, timeline: Timeline, event: str, tag: str = ""):
        self._timeline = timeline
        self._event = event
        self._tag = tag
        self._start_time_ms = 0.0
        self._end_time_ms = 0.0

    def _enter(self):
        self._start_time_ms = time.time() * 1000.0

    def _exit(self):
        self._end_time_ms = time.time() * 1000.0
        timeline_event = TimelineEvent(
            event=self._event,
            start_time_ms=self._start_time_ms,
            end_time_ms=self._end_time_ms,
            tag=self._tag,
        )
        self._timeline.add_event(timeline_event)

    def __enter__(self):
        return self._enter()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._exit()

    async def __aenter__(self):
        return self._enter()

    def __aexit__(self, exc_type, exc_value, exc_tb):
        self._exit()
