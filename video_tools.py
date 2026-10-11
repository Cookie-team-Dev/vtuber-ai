import os

def trim_video(input_path, output_path, start=0, end=None):
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_path)
    if end is None:
        end = clip.duration
    trimmed = clip.subclipped(start, end)
    trimmed.write_videofile(output_path, logger=None)
    trimmed.close()
    clip.close()
    return output_path

def speed_up_video(input_path, output_path, speed=2.0):
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_path)
    sped = clip.with_speed_scaled(speed)
    sped.write_videofile(output_path, logger=None)
    sped.close()
    clip.close()
    return output_path

def slow_down_video(input_path, output_path, speed=0.5):
    return speed_up_video(input_path, output_path, speed)

def crop_video(input_path, output_path, x1=0, y1=0, x2=None, y2=None):
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_path)
    if x2 is None:
        x2 = clip.size[0]
    if y2 is None:
        y2 = clip.size[1]
    cropped = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    cropped.write_videofile(output_path, logger=None)
    cropped.close()
    clip.close()
    return output_path

def resize_video(input_path, output_path, width=None, height=None):
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_path)
    if width is not None and height is not None:
        resized = clip.resized((width, height))
    elif width is not None:
        resized = clip.resized(width=width)
    elif height is not None:
        resized = clip.resized(height=height)
    else:
        resized = clip
    resized.write_videofile(output_path, logger=None)
    resized.close()
    clip.close()
    return output_path

def extract_audio(input_path, output_path):
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_path)
    if clip.audio is not None:
        clip.audio.write_audiofile(output_path, logger=None)
    clip.close()
    if os.path.exists(output_path):
        return output_path
    return None

def mute_video(input_path, output_path):
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_path)
    muted = clip.without_audio()
    muted.write_videofile(output_path, logger=None)
    muted.close()
    clip.close()
    return output_path

def get_video_info(input_path):
    from moviepy import VideoFileClip
    try:
        clip = VideoFileClip(input_path)
        info = {
            'duration': clip.duration,
            'fps': clip.fps,
            'size': clip.size,
            'has_audio': clip.audio is not None,
            'format': os.path.splitext(input_path)[1],
        }
        clip.close()
        return info
    except Exception as e:
        return {'error': str(e)}

def gif_to_video(gif_path, output_path=None):
    from moviepy import VideoFileClip
    if output_path is None:
        base, _ = os.path.splitext(gif_path)
        output_path = f"{base}.mp4"
    clip = VideoFileClip(gif_path)
    clip.write_videofile(output_path, logger=None)
    clip.close()
    return output_path

def video_to_gif(input_path, output_path=None, start=0, end=None, fps=15):
    from moviepy import VideoFileClip
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.gif"
    clip = VideoFileClip(input_path)
    if end is None:
        end = clip.duration
    subclip = clip.subclipped(start, end)
    subclip.write_gif(output_path, fps=fps, logger=None)
    subclip.close()
    clip.close()
    return output_path
