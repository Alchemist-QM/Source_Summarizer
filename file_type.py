from pathlib import Path

def detect_file_type(file_path: str) -> str:
    """
    Determines the file type category based on file extension.

    Returns:
        "audio", "video", "pdf", "text", "markdown", "image", or "unknown"
    """
    ext = Path(file_path).suffix.lower()

    audio_exts = {".mp3", ".wav", }
    video_exts = {".mp4",".webm"}
    md_exts    = {".md", ".markdown", }
    pdf_exts   = {".pdf"}
    docx = {".docx", ".doc"}
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

    if ext in audio_exts:
        return "audio"
    elif ext in video_exts:
        return "video"
    elif ext in pdf_exts:
        return "pdf"
    elif ext in md_exts:
        return "markdown"
    elif ext in image_exts:
        return "image"
    elif ext in docx:
        return "docx"
    else:
        return "unknown"
    
    
if __name__ == "__main__":
    test_files = [
        "song.mp3",
        "movie.mp4",
        "document.pdf",
        "notes.md",
        "picture.jpg",
        "archive.zip"
    ]
    audio_file =[ "C:/Users/qmerr/Downloads/summarizer/02 DEMO ICMP Sweeps with Angry IP and Nmap.mp4",
        "C:/Users/qmerr/Downloads/summarizer/02 DEMO ICMP Sweeps with Angry IP and Nmap.wav",
        "C:/Users/qmerr/Downloads/summarizer/outputs/chunks/53e0befa_09-32-905_chunks.md",
        "C:/Users/qmerr/Downloads/summarizer/Network Scanning.pdf",
        "C:/Users/qmerr/Downloads/summarizer/test_audio.wav",
        "C:/Users/qmerr/Downloads/summarizer/Scanning Network.docx"
        ]
    for file in test_files:
        file_type = detect_file_type(file)
        print(f"{file}: {file_type}")
    for file in audio_file:
        file_type = detect_file_type(file)
        print(f"{file_type}")