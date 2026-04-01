import os, time, logging, shutil
import subprocess, yt_dlp, json, asyncio 
from functools import wraps 
from typing import Callable, Any, Coroutine, List, Dict
from faster_whisper import WhisperModel, BatchedInferencePipeline
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from datetime import timedelta
from dotenv import load_dotenv
from youtube_vid_info import Video_Info
from prompts import youtube_prompt_templates
from token_batching import TokenSplitter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import tiktoken

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #temporary fix 

vid_info = Video_Info()
model = WhisperModel("base", device="cpu",compute_type="int8", num_workers=4)
pipeline = BatchedInferencePipeline(model=model)

load_dotenv()
fast_llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY_3"),
    model='gpt-3.5-turbo'
    )


def is_youtube_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    if "youtube.com/watch?v=" in url:
        return True
    return ValueError("Invalid YouTube URL")


def log_timer(label: str):
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"{label} completed in {duration:.2f}second ")
            return result
        return wrapper
    return decorator

def download_video(url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_options = {
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'format': 'bv*[height<=720]+ba/b[height<=720]',
            'merge_output_format': 'mp4',
            'retries': 10,
            'noprogress': True,
            'http_chunk_size': 10485760,
            'quiet': False,
        }
    try:
        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            return ydl.extract_info(url, download=True)
    except Exception as e:
        return f"Error with downloading: {e}"

def extract_frames(video_path, frames_dir, duration_seconds, target_frame_count=50):
    os.makedirs(frames_dir, exist_ok=True)
    interval_sec = max(duration_seconds / target_frame_count, 1)
    cmd = [
        'ffmpeg',
        '-threads', 'auto',
        '-i', video_path,
        '-an',
        '-vf', f'fps=1/{interval_sec},scale=iw/2:ih/2',
        '-qscale:v', '5',
        os.path.join(frames_dir, 'frame_%06d.jpg')
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return interval_sec

def split_video_to_audio_chunks(video_path, output_dir, chunk_length_sec=30):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "ffmpeg", "-i", video_path,
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-ar", "16000",
        "-ac", "1",
        os.path.join(output_dir, "chunk_%03d.wav")
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chunk_paths = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.startswith("chunk_") and f.endswith(".wav")
    ])
    return chunk_paths

def format_timestamp(seconds: float) -> str:
    delta = timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    milliseconds = int((seconds - total_seconds) * 1000)
    formatted = str(timedelta(seconds=total_seconds)) + f".{milliseconds:03d}"
    return formatted

def transcribe_chunk(audio_path, offset, pipeline)->list:
    results, _ = pipeline.transcribe(
        audio_path,
        chunk_length=30,
        vad_filter=True,
    )
    segments = []
    for result in results:
        adjusted_start = offset + int(result.start)
        adjusted_end = offset + int(result.end)
        segments.append({
            "start": adjusted_start,
            "end": adjusted_end,
            "text": result.text,
            "timestamp": f"{format_timestamp(adjusted_start)}--> {format_timestamp(adjusted_end)}"
        })
    return segments

def transcribe_audio_chunks(chunk_paths, pipeline):
    all_segments = []
    chunk_duration_sec = 30
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(transcribe_chunk, audio_path, i * chunk_duration_sec, pipeline)
            for i, audio_path in enumerate(chunk_paths)
        ]
        for future in futures:
            all_segments.extend(future.result())
    return all_segments

def match_frames_to_transcript(segments: list, frames_dir: str, interval_sec: float = 1.0) -> list:
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: float(s["start"]))
    results = []
    seg_idx = 0
    num_segments = len(segments)
    for frame_file in frame_files:
        try:
            frame_idx = int(frame_file.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        frame_time = frame_idx * interval_sec
        matched_texts = []
        while seg_idx < num_segments:
            seg = segments[seg_idx]
            try:
                start = float(seg["start"])
                end = float(seg["end"])
            except (KeyError, ValueError, TypeError):
                seg_idx += 1
                continue
            if end <= frame_time:
                seg_idx += 1
                continue
            if start <= frame_time < end:
                matched_texts.append({
                    "text": seg["text"],
                    "timestamp": seg["timestamp"]
                })
            elif start > frame_time:
                break
            seg_idx += 1
        if matched_texts:
            results.append({
                "frame": os.path.join(frames_dir, frame_file).replace("\\", "/"),
                "texts": matched_texts,
                "frame_time": f"{frame_time:.2f}"
            })
    return results
def save_md_transcript(transcribed_segments, md_path="md.transcript.md"):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Video Transcript\n\n")
        for seg in transcribed_segments:
            ts = seg.get("timestamp", "")
            text = seg.get("text", "")
            f.write(f"- **{ts}**\n\n    {text}\n\n")
            
            
def choose_summarization_strategy(duration=None, duration_threshold=600):
    if duration is None:
        return "map_reduce"
    return "single_pass" if duration <= duration_threshold else "map_reduce"

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

@log_timer("Downloading video")
async def timed_download_video(url: str, output_dir: str = "downloads"):
    return await asyncio.to_thread(download_video, url, output_dir)

@log_timer("Frame extraction")
async def timed_extract_frames(video_path, frames_dir, duration):
    return await asyncio.to_thread(extract_frames, video_path, frames_dir, duration)

@log_timer("Audio chunking")
async def timed_audio_split(video_path, audio_chunks_dir):
    return await asyncio.to_thread(split_video_to_audio_chunks, video_path, audio_chunks_dir)

@log_timer("Audio transcription")
async def timed_transcription(chunk_paths, pipeline):
    return await asyncio.to_thread(transcribe_audio_chunks, chunk_paths, pipeline)

@log_timer("Frame-to-transcript matching")
async def timed_match_frames(segments, frames_dir, interval_sec):
    return match_frames_to_transcript(segments, frames_dir, interval_sec)

@log_timer("Chunk summarization")
async def timed_summary(segment_texts, strategy, **chains):
    return await new_summarize_chunks_token_batching(segment_texts, **chains)

def create_timeline(
    frame_matches: list,
    chunk_summaries: list,
    chunk_duration: int = 30
) -> list:
    frames_df = pd.DataFrame(frame_matches)
    chunks_df = pd.DataFrame(chunk_summaries)
    if frames_df.empty or chunks_df.empty:
        logging.warning("⚠️ Empty DataFrame detected: frames_df or chunks_df is empty.")
        return []
    required_cols = ['chunk_index', 'summary', 'insights']
    for col in required_cols:
        if col not in chunks_df.columns:
            chunks_df[col] = None
    frames_df['frame_time'] = pd.to_numeric(frames_df['frame_time'], errors='coerce').fillna(0.0)
    frames_df['chunk_index'] = (frames_df['frame_time'] // chunk_duration).astype(int)
    merged_df = pd.merge(frames_df, chunks_df, on='chunk_index', how='left')
    timeline = []
    for _, row in merged_df.iterrows():
        try:
            texts = row.get("texts", [])
            timeline.append({
                "frame": row['frame'],
                "frame_time": f"{row['frame_time']:.2f}",
                "transcript": [t.get("text", "") for t in texts] if isinstance(texts, list) else [],
                "timestamp": [t.get("timestamp", "") for t in texts] if isinstance(texts, list) else [],
            })
        except Exception as e:
            continue
    return timeline

def clear_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.path(folder_path).mkdir()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# PATCH: Efficient summarization using token batching
async def new_summarize_chunks_token_batching(
    chunks: List[str],
    summary_chain: Runnable,
    insight_chain: Runnable,
    reduce_summary_chain: Runnable,
    reduce_insight_chain: Runnable,
    model_name: str = "gpt-3.5-turbo"
) -> Dict:
    tokenizer = TokenSplitter.get_tokenizer(model_name)
    model_max_tokens = TokenSplitter.get_max_tokens_for_model(model_name)
    chunk_token_limit = min(2000, int(model_max_tokens * 0.25))
    combined_text = "\n".join(chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_token_limit,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_chunks = splitter.split_text(combined_text)
    batches = TokenSplitter.tokenize_and_batch(split_chunks, max_batch_tokens=model_max_tokens//2, tokenizer=tokenizer, use_splitter=False)
    summaries, insights = [], []
    per_chunk_data = []
    for batch in batches:
        batch_texts = [{"text": str(text)} for text in batch]
        summary_batch = await summary_chain.abatch(batch_texts)
        insight_batch = await insight_chain.abatch(batch_texts)
        for idx, (summary_res, insight_res, chunk_text) in enumerate(zip(summary_batch, insight_batch, batch)):
            chunk_summary = TokenSplitter.safe_content(summary_res).strip()
            chunk_insights = TokenSplitter.safe_content(insight_res).strip()
            summaries.append(chunk_summary)
            insights.append(chunk_insights)
            per_chunk_data.append({
                "chunk_index": len(per_chunk_data),
                "text": chunk_text,
                "summary": chunk_summary,
                "insights": chunk_insights
            })
    combined_summary_text = "\n".join(summaries)
    combined_insight_text = "\n".join(insights)
    final_summary = (await reduce_summary_chain.ainvoke({"text": combined_summary_text})).content.strip()
    final_insights = (await reduce_insight_chain.ainvoke({"text": combined_insight_text})).content.strip()
    return {
        "final_summary": final_summary,
        "final_insights": final_insights,
        "per_chunk": per_chunk_data
    }

async def youtube_summarizer_tunnel(url:str, prompt_templates:dict):
    start_time = time.time()
    output_dir = "downloads"
    frames_dir = "frames"
    audio_chunks_dir = "temp_chunks"

    clear_folder(frames_dir)
    clear_folder(output_dir)
    clear_folder(audio_chunks_dir)

    map_llm= fast_llm
    summary_chain = prompt_templates['summary'] | map_llm
    insight_chain= prompt_templates['insights'] | map_llm
    reduce_summary_chain = prompt_templates['default'] | map_llm
    reduce_insight_chain = prompt_templates['final_insight'] | map_llm

    tldr_chain = prompt_templates['tldr'] | fast_llm
    introduction_chain = prompt_templates['introduction'] | fast_llm
    takeaway_chain = prompt_templates['takeaway'] | fast_llm
    conclusion_chain = prompt_templates['final_conclusion'] | fast_llm

    info = await timed_download_video(url, output_dir=output_dir)
    #add logic for not vid info 
    video_info = {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "webpage_url": info.get("webpage_url"),
        "upload_date": info.get("upload_date"),
    }
    video_path = next(
        (os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")),
        None
    )
    if not video_path:
        logging.error("Video Path not found.")
        return
    duration = vid_info.get_video_duration(video_path, url=url)
    strategy = choose_summarization_strategy(duration)
    interval_sec = await timed_extract_frames(video_path, frames_dir, duration)
    chunk_paths = await timed_audio_split(video_path, audio_chunks_dir)
    transcribed_segments = await timed_transcription(chunk_paths, pipeline)
    save_md_transcript(transcribed_segments, md_path="md.transcript.md")

    for path in chunk_paths:
        os.remove(path)
    frame_match = await timed_match_frames(transcribed_segments, frames_dir, interval_sec)
    segment_texts = [seg['text'] for seg in transcribed_segments]
    summary = await timed_summary(
        segment_texts,
        strategy,
        summary_chain=summary_chain,
        insight_chain=insight_chain,
        reduce_summary_chain=reduce_summary_chain,
        reduce_insight_chain=reduce_insight_chain
    )
    timeline = await asyncio.to_thread(
        create_timeline,
        frame_matches=frame_match, 
        chunk_summaries=summary.get("per_chunk", []),
    )
    all_summaries_text = summary.get('final_summary', '')
    per_chunk_summaries = summary.get('per_chunk', [])
    all_takeaways_text = "\n".join(chunk.get('insights', '') for chunk in per_chunk_summaries)
    user_query = f"Summarization of video: {video_info.get('title', '')}"

    tldr = (await tldr_chain.ainvoke({"all_summaries": all_summaries_text})).content.strip()
    introduction = (await introduction_chain.ainvoke({
        "user_query": user_query,
        "all_summaries": all_summaries_text
    })).content.strip()
    takeaways = (await takeaway_chain.ainvoke({"takeaways": all_takeaways_text})).content.strip()
    conclusion = (await conclusion_chain.ainvoke({"all_summaries": all_summaries_text})).content.strip()

    # PATCH: final_summary and final_insights appended at the top-level, not in timeline
    with open("final_output.json", "w", encoding="utf-8") as f:
        json.dump({
            "video_info": video_info,
            "final_summary": summary.get('final_summary', ''),
            "final_insights": summary.get('final_insights', ''),
            "timeline": timeline,
            "frame_match": frame_match,
            "tldr": tldr,
            "introduction": introduction,
            "takeaways": takeaways,
            "conclusion": conclusion
        }, f, ensure_ascii=False, indent=2)

    logging.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    user_video = input("Please insert a url: ")
    asyncio.run(youtube_summarizer_tunnel(url=user_video,prompt_templates=prompt_templates))
