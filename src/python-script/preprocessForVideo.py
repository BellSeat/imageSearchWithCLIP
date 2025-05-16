import subprocess
import os
import cv2
import datetime
import shlex
from log_util import setup_local_logger
logger = setup_local_logger()
ffmpegLogger = setup_local_logger('logs/ffempg.log')

# check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False
    
# extract keyframes from video
def get_Pframe_timestamps(video_path):
    cmd = (
        f"ffprobe -select_streams v -show_frames -show_entries "
        f"frame=pict_type,best_effort_timestamp_time,coded_picture_number -of csv {shlex.quote(video_path)}"
    )
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    timestamps = []
    for line in result.stdout.splitlines():
        # logger.info(line)
        parts = line.split(',')
        i = 0
        if len(parts) >= 3 and parts[0] == 'frame' and parts[2] in ['I']:
            # logger.info(parts)

            try: 
                timestamp = float(parts[1])
                timestamps.append(timestamp)
            except ValueError:
                continue
    logger.info(f"Extracted {len(timestamps)} keyframes")
    return timestamps

# extract keyframes from video
def extract_keyframmes(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamps = get_Pframe_timestamps(video_path)
    logger.info(f'Timestamps: {timestamps}')
    if not timestamps:
        logger.info("No keyframes found in the video.")
        return
    
    for i,ts in enumerate(timestamps,start=1):
        index_str = f"{i:04d}"
        timestamps_str = format_time(ts)
        output_file = os.path.join(output_dir, f"frame_{index_str}_{timestamps_str}.jpg")

        command = [
            'ffmpeg', '-ss', str(ts), '-i', video_path, '-frames:v', '1', 
            '-q:v', '2', '-update', '1', output_file
        ]

    
        try:
            results = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            ffmpegLogger.info(results.stdout)

        except subprocess.CalledProcessError as e:
            logger.info(f"Error extracting keyframes: {e}")

# name the ouput directory
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.info("Error opening video file")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap.release()
    return total_frames

def build_output_dir(video_path):
    # Get the video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get the current date and time
    now = datetime.datetime.now()
    
    # Get the total number of frames in the video
    total_frames = get_video_info(video_path)
    # Format the date and time
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the output directory name
    dir_name = f"{video_name}_{formatted_date}_frames_{total_frames}"

    # save the output in database/raw/video/frames/
    output_dir = os.path.join("database", "raw", "video", "frames", dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# format the time
def format_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    return str(td).replace(':','-').replace('.','-')
# main function
if __name__ == "__main__":
    # check if the platform is Windows
    import platform
    video_path = ""
    if platform.system() == "Windows":
        video_path = os.path.abspath(r"E:\demo\imageSearchWithCLIP\sample\video\sample.mp4").replace('\\', '/')
    else:
        video_path = r"..\..\sample\video\sample.mp4" 
    output_dir = build_output_dir(video_path)
    if check_ffmpeg():
        extract_keyframmes(video_path, output_dir)
    else:
        logger.info("ffmpeg is not installed. Please install ffmpeg to use this script.")