import subprocess 
import json


class Video_Info:
    def __init__(self):
        pass
    
    def get_video_duration(self, video_path, url=None):
        """ 
        Uses ffprobe to retrieve duration of the video
        
        in seconds
        
        Args:
            video_path(str): path to video file
        
        """
        if video_path:
            command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
            ]
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
                output= json.loads(result.stdout)
                duration = float(output['format']['duration'])
                return duration
            except (subprocess.CalledProcessError, KeyError, ValueError) as e:
                print(f"[ERROR] Could not determine video duration: {e}")
                return None
        
        else:
            command = [
                "yt-dlp",
                "--get-duration",
                url,
            ]
            
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
                duration_str = result.stdout.strip()
                # Convert HH:MM:SS to seconds
                parts = list(map(int, duration_str.split(":")))
                if len(parts) == 3:
                    h, m, s = parts
                elif len(parts) == 2:
                    h = 0
                    m, s = parts
                else:
                    h = m = 0
                    s = parts[0]
                return h * 3600 + m * 60 + s
            except Exception as e:
                print(f"[ERROR] Could not determine video duration (yt-dlp): {e}")
                return None

        
    def get_metadata(self, url:str):
        command = [
            "yt-dlp",
            "-j",
            url,
        ]
        
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            info = json.loads(result.stdout)
            
            metadata = {
                "title": info.get("title"),
            "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration")/60,
            "filesize": info.get("filesize_approx")/1000000 or info.get("filesize"),
            "channel_id": info.get("channel_id"),
            
        }
            return metadata
        except (subprocess.CalledProcessError, KeyError, ValueError) as e:
            print(f"[ERROR] Could not determine video duration: {e}")
            return None
        
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse JSON from yt-dlp output.")
            return None
        
        
    def calculate_dynamic_interval(self, duration_seconds, target_frame_count=45):
        """"   
        Takes duration to extract a set amount of frames per video
        
        """
        
        if duration_seconds and duration_seconds > 0:
            return max(int(duration_seconds // target_frame_count), 1)
        return 5  # fallback