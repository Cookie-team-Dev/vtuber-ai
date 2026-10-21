import os
import shutil
import tempfile
import subprocess

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    VideoFileClip = None
    MOVIEPY_AVAILABLE = False

def _require_moviepy():
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy is not installed")

def trim_video(input_path, output_path, start=0, end=None):
    _require_moviepy()
    clip = None
    trimmed = None
    try:
        clip = VideoFileClip(input_path)
        if end is None:
            end = clip.duration
        trimmed = clip.subclipped(start, end)
        trimmed.write_videofile(output_path, logger=None)
        return output_path
    finally:
        if trimmed is not None:
            trimmed.close()
        if clip is not None:
            clip.close()

def speed_up_video(input_path, output_path, speed=2.0):
    _require_moviepy()
    clip = None
    sped = None
    try:
        clip = VideoFileClip(input_path)
        sped = clip.with_speed_scaled(speed)
        sped.write_videofile(output_path, logger=None)
        return output_path
    finally:
        if sped is not None:
            sped.close()
        if clip is not None:
            clip.close()

def slow_down_video(input_path, output_path, speed=0.5):
    return speed_up_video(input_path, output_path, speed)

def crop_video(input_path, output_path, x1=0, y1=0, x2=None, y2=None):
    _require_moviepy()
    clip = None
    cropped = None
    try:
        clip = VideoFileClip(input_path)
        if x2 is None:
            x2 = clip.size[0]
        if y2 is None:
            y2 = clip.size[1]
        cropped = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
        cropped.write_videofile(output_path, logger=None)
        return output_path
    finally:
        if cropped is not None:
            cropped.close()
        if clip is not None:
            clip.close()

def resize_video(input_path, output_path, width=None, height=None):
    if width is None and height is None:
        shutil.copyfile(input_path, output_path)
        return output_path
    _require_moviepy()
    clip = None
    resized = None
    try:
        clip = VideoFileClip(input_path)
        if width is not None and height is not None:
            resized = clip.resized((width, height))
        elif width is not None:
            resized = clip.resized(width=width)
        elif height is not None:
            resized = clip.resized(height=height)
        resized.write_videofile(output_path, logger=None)
        return output_path
    finally:
        if resized is not None and resized is not clip:
            resized.close()
        if clip is not None:
            clip.close()

def extract_audio(input_path, output_path):
    _require_moviepy()
    clip = None
    try:
        clip = VideoFileClip(input_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(output_path, logger=None)
            audio_ok = True
        else:
            audio_ok = False
        return output_path if audio_ok else None
    finally:
        if clip is not None:
            clip.close()

def mute_video(input_path, output_path):
    _require_moviepy()
    clip = None
    muted = None
    try:
        clip = VideoFileClip(input_path)
        muted = clip.without_audio()
        muted.write_videofile(output_path, logger=None)
        return output_path
    finally:
        if muted is not None:
            muted.close()
        if clip is not None:
            clip.close()

def get_video_info(input_path):
    if not MOVIEPY_AVAILABLE:
        return {'error': 'moviepy is not installed'}
    clip = None
    try:
        clip = VideoFileClip(input_path)
        info = {
            'duration': clip.duration,
            'fps': clip.fps,
            'size': clip.size,
            'has_audio': clip.audio is not None,
            'format': os.path.splitext(input_path)[1],
        }
        return info
    except Exception as e:
        return {'error': str(e)}
    finally:
        if clip is not None:
            clip.close()

def video_watch_description(path, max_frames=10):
    """Describe a video by sampling frames."""
    try:
        import cv2
    except Exception as e:
        return f"cv2 not available: {e}"
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return f"Could not open {path}"
        frames_read = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        cap = None
        info = get_video_info(path)
        if 'error' in info:
            return f"Could not analyze video: {info['error']}"
        return f"Video: {frame_count} frames, {info['duration']:.1f}s, {info['fps']:.1f}fps, {info['size'][0]}x{info['size'][1]}, audio={info['has_audio']}"
    finally:
        if cap is not None:
            cap.release()

def gif_to_video_command(gif_path, output=None):
    return gif_to_video(gif_path, output)

def video_to_gif_command(input_path, output=None, start=0, end=None, fps=15):
    return video_to_gif(input_path, output, float(start), float(end) if end is not None else None, int(fps))

def gif_to_video(gif_path, output_path=None):
    import os
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy is not installed")
    clip = None
    try:
        if output_path is None:
            base, _ = os.path.splitext(gif_path)
            output_path = f"{base}.mp4"
        clip = VideoFileClip(gif_path)
        clip.write_videofile(output_path, logger=None)
        return output_path
    finally:
        if clip is not None:
            clip.close()

def video_to_gif(input_path, output_path=None, start=0, end=None, fps=15):
    import os
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy is not installed")
    clip = None
    subclip = None
    try:
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}.gif"
        clip = VideoFileClip(input_path)
        if end is None:
            end = clip.duration
        subclip = clip.subclipped(start, end)
        subclip.write_gif(output_path, fps=fps, logger=None)
        return output_path
    finally:
        if subclip is not None:
            subclip.close()
        if clip is not None:
            clip.close()
