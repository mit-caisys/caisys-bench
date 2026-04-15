import cv2
from PIL import Image
from pydub import AudioSegment
from pydub.effects import normalize
from scenedetect import FrameTimecode
from scenedetect import open_video
from scenedetect import VideoStream
from io import BytesIO
from scene import Scene
import io
import base64

from typing import List, Optional, Tuple, Dict, Union

DEFAULT_FRAME_EXTRACTION_INTERVAL_S = 5
DEFAULT_SCENE_THRESHOLD = 27
DEFAULT_MIN_SCENE_LEN_S = 1
DEFAULT_NUM_FRAMES = 5


def encode_image(img: Image) -> str:
    output_buffer = BytesIO()
    img.save(output_buffer, format="JPEG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

def get_metadata(
    video_path: str
    ) -> Dict[str, Union[str, int, float]]:
        video = open_video(video_path)
        response = {
            "video_path": video_path,
            "frame_rate": video.frame_rate,
            "duration": video.duration.get_seconds(),
        }
        return response

def get_frames(
        video: VideoStream,
        scene: Scene,
        frame_extraction_interval_s: Optional[
            int
        ] = DEFAULT_FRAME_EXTRACTION_INTERVAL_S,
        interval_num_frames: Optional[int] = None,
    ) -> Tuple[List[Image.Image], List[float]]:
        video.seek(scene.start)
        frames = []
        time_stamps = []
        if interval_num_frames is None:
            interval_num_frames = int(
                frame_extraction_interval_s * video.frame_rate
            )
        scene_len = scene.end.get_frames() - scene.start.get_frames()
        for index in range(scene_len):
            if index % interval_num_frames == 0:
                f = video.read()
                if f is False:
                    continue
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(f)
                frames.append(im)
                time_stamps.append(video.position.get_seconds())
            else:
                video.read(decode=False)
        return frames, time_stamps

def get_num_frames(video_path: str, start_time: float, end_time: float, frame_extraction_interval_s: int):
     video = open_video(video_path)
     frame_rate = video.frame_rate
     video_id = video_path
     
     ft_start = FrameTimecode(timecode=float(start_time), fps=frame_rate)
     ft_end = FrameTimecode(timecode=float(end_time), fps=frame_rate)
        # Get frames
     scene = Scene(start=ft_start, end=ft_end, video_id=video_id)

     frames, _ = get_frames(
            video,
            scene,
            frame_extraction_interval_s 
        )
     return frames


def audio_segment_to_bytes(audio: AudioSegment, key) -> io.BytesIO:
    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="mp3")
    audio_bytes.seek(0)
    audio_bytes.name = f"{key}.mp3"
    return audio_bytes

def extract_audio(video_path: str) -> List[AudioSegment]:
    audio_clips = []
    try:
        audio = AudioSegment.from_file(video_path)
        audio = normalize(audio)
        audio_clips.append(audio)
    except Exception:
        return audio_clips

    return audio_clips

def get_frames_and_audio_in_range(
    video_path: str,
    start_time: float,
    end_time: float,
    frame_extraction_interval_s: int,
) -> Tuple[List[str], str]:
    # logger.debug(f"Extracting: {video_path}, {start_time}s to {end_time}s")
    video_id = video_path
    video = open_video(video_path)
    frame_rate = video.frame_rate
    if start_time == end_time:
        end_time += 1
    ft_start = FrameTimecode(timecode=float(start_time), fps=frame_rate)
    ft_end = FrameTimecode(timecode=float(end_time), fps=frame_rate)
    # Get frames
    scene = Scene(start=ft_start, end=ft_end)
    frames, _ = get_frames(
        video,
        scene,
        frame_extraction_interval_s=frame_extraction_interval_s,
    )

    audio_clips = extract_audio(video_path)
    return frames, audio_clips