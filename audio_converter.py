#!/usr/bin/env python3
"""
Audio → Token-Chunked Markdown converter using faster-whisper
Outputs a single markdown file chunked by tokens (LLM-safe)
"""
import hashlib
from pathlib import Path
from datetime import timedelta, datetime
from typing import Optional, List, Union
from faster_whisper import WhisperModel, BatchedInferencePipeline
from langchain.schema import Document
from token_batching import TokenSplitter
from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = {
    ".pptx", ".csv", ".docx",
    ".png", ".jpg", ".jpeg",
    ".pdf", ".webp", ".vtt", ".md",
    
}

def file_convertor(
    file_path:Union[str, List[str]],
    converter: DocumentConverter, 
    output_dir: str = "outputs", 
    debug:bool=False,
    ) -> Union[Union[str, tuple[str, dict]], tuple[List[str], List[str]]]:
    """
    Converts pptx, csv, docx, png, jpeg

    Args:
    file_path (str): Path to input file
    output_dir (str): Directory to save markdown output

    Returns:
        tuple[str, dict]: Markdown content and document info dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        result = converter.convert(str(file_path))

        markdown_text = result.document.export_to_markdown()
        if debug:
            doc_info = {
                'file_name': file_path.name,
                'format': file_path.suffix,
                'status': 'success',
                'markdown_length': len(markdown_text),
            }
        output_file = output_dir / f"{file_path.stem}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        if debug:       
            doc_info['output_file'] = str(output_file)
        return output_file, doc_info if debug else output_file
    
    if isinstance(file_path, list):
        md_results = []
        md_file_paths = []
        for path in file_path: 
            if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {Path(path).suffix}")
            
        results = converter.convert_all(file_path)
        for res in results:
            markdown_text = res.document.export_to_markdown()
            md_results.append(markdown_text)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_hash = hashlib.md5(str(res).encode()).hexdigest()[:8]
            
            output_file = output_dir / f"{result_hash}_{timestamp}.md"
            md_file_paths.append(output_file)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_text)
        return md_results, md_file_paths
    
def format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)

    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def md_timestamp(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return f"**[{format_timestamp(ts)}]** "


# ────────────────────────────────────────────────
# Paragraph Formatter (no chunking yet)
# ────────────────────────────────────────────────
class MarkdownFormatter:
    def __init__(self,
                show_timestamps: bool = False,
                min_words_per_paragraph: int = 25,
                language: str = "en",
                ):
        
        self.show_timestamps = show_timestamps
        self.min_words_per_paragraph = min_words_per_paragraph
        self.language = language
        self.buffer = []
        self.buffer_word_count = 0
        self.paragraphs: List[str] = []
        
    def add_segment(self, segment):
        text = segment.text.strip()
        if not text:
            return

        speaker = getattr(segment, "speaker", None)
        ts_start = segment.start if self.show_timestamps else None

        if speaker !=  self.buffer:
            self.flush_buffer()


        self.buffer.append((ts_start, text))
        self.buffer_word_count += len(text.split())

        if self.buffer_word_count >= self.min_words_per_paragraph:
            self.flush_buffer()

    def flush_buffer(self):
        if not self.buffer:
            return

        lines = []
        for ts, text in self.buffer:
            line = ""
            if self.show_timestamps and ts is not None:
                line += md_timestamp(ts)
            line += text
            lines.append(line)

        paragraph = " ".join(lines)
        self.paragraphs.append(paragraph)

        self.buffer.clear()
        self.buffer_word_count = 0

    def finalize(self) -> List[str]:
        self.flush_buffer()
        return self.paragraphs


# ────────────────────────────────────────────────
# Main function
# ────────────────────────────────────────────────
def audio_to_markdown_token_chunked(
        audio_path:str,
        device: str = "cpu",
        language: str = "en",
        model_size:str = 'base',
        show_timestamps: bool = False,
        min_words_per_paragraph: int = 25,
        vad_filter: bool = True,
        output_dir:str = "outputs/chunks",
        debug: bool = False,
        model_name_for_tokenizer: str = "gpt-4o-mini",
    ):

        model = WhisperModel(
        model_size,
        device=device,
        compute_type="int8",
        num_workers=4,
    ) if device == "cpu" else WhisperModel(
        model_size=model_size,
        device="cuda",
        compute_type="float16",
        num_workers=8,
    )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if vad_filter:
            model = BatchedInferencePipeline(model=model)

        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            without_timestamps=False,   
        )

        formatter = MarkdownFormatter(
            show_timestamps=show_timestamps,
            min_words_per_paragraph=min_words_per_paragraph,
        )

        for segment in segments:
            formatter.add_segment(segment)

        paragraphs = formatter.finalize()

        # Convert paragraphs into LangChain Documents
        docs = [Document(page_content=p) for p in paragraphs]

        # Token splitter
        splitter = TokenSplitter()
        tokenizer = splitter.get_tokenizer(model_name_for_tokenizer)
        max_tokens = splitter.get_max_tokens_for_model(model_name_for_tokenizer)

        batches = splitter.tokenize_and_batch(
        items=docs,
        max_batch_tokens=max_tokens,
        tokenizer=tokenizer,
        use_splitter=True
    )

        
        file_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        timestamp = format_timestamp(info.duration).replace(":", "-").replace(".", "-")
        
        chunked_md_path = output_path / f"{file_hash}_{timestamp}_chunks.md"
        md_lines = []
        for i, batch in enumerate(batches, 1):
            md_lines.append(f"## Chunk {i}\n\n")
            for doc in batch:
                md_lines.append(doc.page_content + "\n\n")
        
        meta_data ={
            "file_name": Path(audio_path).name,
            "language": info.language,
            "language_probability": f"{info.language_probability:.2%}",
            "duration_seconds": f"{info.duration:.1f}",
        }

        if debug == True:
            md_lines.append("\n\n---\n\n")
            md_lines.append(str(meta_data))
            
        with open(chunked_md_path, "w", encoding="utf-8") as f:
            f.writelines(md_lines)
            
        return chunked_md_path, md_lines


# ────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────
"""
if __name__ == "__main__":
    #audio_file = "C:/Users/qmerr/Downloads/summarizer/02 DEMO ICMP Sweeps with Angry IP and Nmap.mp4"
    #chunked_path, md_lines = audio_to_markdown_token_chunked(audio_file)
    #print(md_lines)
    
    test_files = ["C:/Users/qmerr/Downloads/summarizer/test_files/Job_Interview.pptx",
                "C:/Users/qmerr/Downloads/summarizer/test_files/kg_test.png",
                "C:/Users/qmerr/Downloads/summarizer/test_files/mamba_test.webp",
               "C:/Users/qmerr/Downloads/summarizer/test_files/The Master Algorithm .docx"        
    ]
    
    markdown_text = file_convertor(test_files, DocumentConverter())
"""