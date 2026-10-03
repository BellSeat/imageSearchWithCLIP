import subprocess
import subprocess
import os
import json
import cv2
import datetime
import shlex
import time
import uuid
from .log_util import setup_local_logger

logger = setup_local_logger()
# Corrected typo: ffempg.log to ffmpeg.log
ffmpegLogger = setup_local_logger('logs/ffmpeg.log')


class VideoProcessingCancelled(Exception):
    """Raised when a background video job is cancelled."""


def wait_for_processing(resume_event=None, cancel_event=None):
    while resume_event is not None and not resume_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            raise VideoProcessingCancelled()
        time.sleep(0.2)
    if cancel_event is not None and cancel_event.is_set():
        raise VideoProcessingCancelled()

# check if ffmpeg is installed


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        logger.error(
            "ffmpeg is not installed. Please install ffmpeg to use video processing features.")
        return False

# extract keyframes from video


def get_keyframe_timestamps(video_path: str, cancel_event=None) -> list:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_frames",
        "-show_entries", "frame=pict_type,best_effort_timestamp_time",
        "-of", "json",
        video_path
    ]
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise VideoProcessingCancelled()
        result = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe error: {result.stderr}")
        timestamps = []
        probe_data = json.loads(result.stdout or "{}")
        for frame in probe_data.get("frames", []):
            if frame.get("pict_type") != "I":
                continue
            timestamp_value = frame.get("best_effort_timestamp_time")
            if timestamp_value is None:
                continue
            try:
                timestamps.append(float(timestamp_value))
            except (TypeError, ValueError):
                continue
        logger.info(f"Extracted {len(timestamps)} keyframes from {video_path}")
        return timestamps
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffprobe: {e.stderr}", exc_info=True)
        return []

# extract keyframes from video and return paths


# Renamed to extract_keyframes, returns paths
def extract_keyframes(video_path, output_dir, cancel_event=None, resume_event=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is not installed, cannot extract keyframes.")

    timestamps = get_keyframe_timestamps(video_path, cancel_event)
    logger.info(f'Keyframe Timestamps: {timestamps}')
    if not timestamps:
        logger.info(f"No keyframes found in video: {video_path}.")
        return []

    extracted_frame_paths = []
    for i, ts in enumerate(timestamps, start=1):
        wait_for_processing(resume_event, cancel_event)
        index_str = f"{i:04d}"
        timestamps_str = format_time(ts)  # Assuming format_time is defined
        output_file = os.path.join(
            output_dir, f"frame_{index_str}_{timestamps_str}.jpg")

        command = [
            'ffmpeg', '-ss', str(ts), '-i', video_path, '-frames:v', '1',
            '-q:v', '2', '-update', '1', output_file
        ]

        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    process.terminate()
                    process.wait(timeout=5)
                    raise VideoProcessingCancelled()
                time.sleep(0.1)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, command, output=stdout, stderr=stderr)
            ffmpegLogger.info(
                f"FFmpeg output for {output_file}: {stdout.strip()}")
            if stderr:  # Log stderr for warnings/non-fatal errors
                ffmpegLogger.warning(
                    f"FFmpeg stderr for {output_file}: {stderr.strip()}")
            extracted_frame_paths.append(output_file)
        except VideoProcessingCancelled:
            raise
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error extracting keyframe {output_file}: {e.stderr}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Unexpected error during keyframe extraction for {output_file}: {e}", exc_info=True)
    logger.info(
        f"Finished extracting {len(extracted_frame_paths)} keyframes to {output_dir}")
    return extracted_frame_paths

# name the ouput directory


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def build_output_dir(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    now = datetime.datetime.now()
    # total_frames could be None if video fails to open
    total_frames = get_video_info(video_path)

    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    dir_name = (
        f"{video_name}_{formatted_date}_{uuid.uuid4().hex[:8]}_frames_"
        f"{total_frames if total_frames is not None else 'unknown'}"
    )
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(source_root, "database", "raw", "video", "frames", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Built output directory for video frames: {output_dir}")
    return output_dir

# format the time


def format_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    # Format as HH-MM-SS-ms (e.g., 0-00-01-501500)
    minutes, seconds_rem = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds
    return f"{hours:01d}-{minutes:02d}-{seconds_rem:02d}-{milliseconds:06d}"

# --- New Function for Video Clipping ---


def clip_video_segment(original_video_path: str, timestamp_s: float, output_path: str, duration_s: int = 4):
    """
    Clips a video segment of a given duration around a specific timestamp.
    Corrected for shell=True and variable name errors.
    """
    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is not installed, cannot clip video segments.")

    # Get video duration using ffprobe
    cmd_duration = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        original_video_path
    ]
    try:
        # FIX: Removed shell=True for security and reliability.
        result = subprocess.run(
            cmd_duration, capture_output=True, text=True, check=True)
        video_duration = float(result.stdout.strip())
    except Exception as e:
        logger.error(
            f"Could not get video duration for {original_video_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get video duration: {e}")

    start_clip = max(0, timestamp_s - (duration_s / 2))

    if start_clip + duration_s > video_duration:
        start_clip = max(0, video_duration - duration_s)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ffmpeg command to clip video
    command = [
        "ffmpeg",
        "-ss", str(start_clip),
        # FIX: Corrected variable name. 'video_path' changed to 'original_video_path'
        "-i", original_video_path,
        # FIX: Corrected variable name. 'duration' changed to 'duration_s'
        "-t", str(duration_s),
        "-c", "copy",       # Fast re-muxing without re-encoding
        "-y",               # Overwrite output file if it exists
        output_path
    ]

    # FIX: Corrected variable names in the log message.
    logger.info(
        f"Clipping video {original_video_path} from {start_clip:.2f}s for {duration_s:.2f}s to {output_path}")
    try:
        results = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # FIX: Changed 'ffmpegLogger' to the standard 'logger'
        logger.info(f"FFmpeg clip output: {results.stdout.strip()}")
        if results.stderr:
            logger.warning(f"FFmpeg clip stderr: {results.stderr.strip()}")
        logger.info(f"Successfully clipped video segment to {output_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error clipping video: {e.stderr}", exc_info=True)
        raise RuntimeError(f"FFmpeg video clipping failed: {e.stderr}")
    except Exception as e:
        logger.error(
            f"Unexpected error during video clipping: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during video clipping: {e}")
