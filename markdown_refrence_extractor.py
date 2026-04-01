import re, json
import pdfx
import re
from datetime import datetime



#use this file to collect the refrences from the markdown file
def extract_references_from_markdown(md_path):
    """
    Extracts the References section from a markdown file.

    Args:
        md_path (str): Path to the markdown file.

    Returns:
        str: Raw text of the References section (may include all lines after '**References**')
    """
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex: find "**References**" and capture everything after it
    match = re.search(r'\*\*References\*\*(.*)', text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        references_text = match.group(1).strip()
        return references_text
    return ""

def split_references(references_text):
    """
    Splits the References section into individual references, assuming each starts with [number].

    Args:
        references_text (str): Raw text from the References section.

    Returns:
        List[str]: List of individual references.
    """
    # Each reference starts with [number]
    refs = re.split(r'\n\[\d+\]\s*', references_text)
    # First element may be empty (before first split)
    return [r.strip() for r in refs if r.strip()]

def parse_reference(ref):
    """
    Attempts to parse an individual reference into fields:
    authors, title, venue, year, arXiv ID, DOI

    Args:
        ref (str): Reference string.

    Returns:
        dict: Parsed fields.
    """
    # Authors: everything before the first quote or dot
    author_match = re.match(r'^([^\"]+?),', ref)
    authors = author_match.group(1).strip() if author_match else ""

    # Year: 4 digit number
    year_match = re.search(r'(\b20\d{2}\b|\b19\d{2}\b)', ref)
    year = year_match.group(1) if year_match else ""

    return {
        "authors": authors,
        "year": year,
        "original": ref
    }


def references_to_json(references_list):
    """
    Converts a list of references to a JSON array with parsed fields.

    Args:
        references_list (List[str]): List of references.

    Returns:
        str: JSON array string.
    """
    data = []
    for i, ref in enumerate(references_list):
        parsed = parse_reference(ref)
        parsed["number"] = i+1
        data.append(parsed)
    return json.dumps(data, indent=2, ensure_ascii=False)

def process_reference_obj(ref_obj):
    """
    Convert a pdfx Reference object to a readable dictionary if possible.
    """
    if hasattr(ref_obj, "as_dict"):
        return ref_obj.as_dict()
    elif hasattr(ref_obj, "url"):
        return {"url": ref_obj.url}
    elif hasattr(ref_obj, "text"):
        return {"text": ref_obj.text}
    else:
        return str(ref_obj)

def format_pdf_date(pdf_date):
    """
    Converts a PDF date string like "D:20101001154705-07'00'" to ISO8601 "2010-10-01T15:47:05-07:00"
    Returns empty string if parsing fails.
    """
    if not pdf_date or not pdf_date.startswith("D:"):
        return ""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-]\d{2})'?(\d{2})'?", pdf_date)
    if match:
        year, month, day, hour, minute, second, tz_hour, tz_minute = match.groups()
        try:
            dt = datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second)
            )
            # Format timezone
            tz = f"{tz_hour}:{tz_minute}"
            return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}{tz}"
        except Exception:
            pass
    # Fallback: just date without timezone
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", pdf_date)
    if match:
        year, month, day, hour, minute, second = match.groups()
        try:
            dt = datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second)
            )
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            pass
    return ""

def clean_title_author(val):
    """
    Adds spaces between capitalized names, e.g. AlexJ.SmolaandVishyS.V.N.Vishwanathan -> Alex J. Smola and Vishy S. V. N. Vishwanathan
    Also fixes missing spaces in title.
    """
    # Add space before capital letter that follows lowercase or period
    val = re.sub(r'(?<=[a-z.])([A-Z])', r' \1', val)
    # Add space after period if missing
    val = re.sub(r'\.([A-Z])', r'. \1', val)
    # Fix "and" (no space around)
    val = re.sub(r'and([A-Z])', r'and \1', val)
    return val.strip()

def extract_metadata_from_pdf(pdf_path) -> dict:
    """
    Extract metadata from a PDF file using the pdfx library.
    Returns: cleaned metadata dictionary
    """
    try:
        pdf = pdfx.PDFx(pdf_path)
        metadata = pdf.get_metadata()
        # Clean author
        metadata['Author'] = clean_title_author(metadata.get('Author', ''))
        # Format creation/modification dates
        metadata['CreationDate'] = format_pdf_date(metadata.get('CreationDate', ''))
        metadata['ModDate'] = format_pdf_date(metadata.get('ModDate', ''))
        return metadata
    except Exception as e:
        print(f"Error extracting citations: {e}")
        return {}



def all_refrence_data(metadata, json_refernces): #make into main reference function to call 
    """
    Converts the citations dictionary to a JSON string.
    """
    return json.dumps({
        "metadata": metadata,
        "references": json_refernces
    }, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    md_path = "converted_markdown.md"
    refs_text = extract_references_from_markdown(md_path)
    all_refs = split_references(refs_text)
    json_output = references_to_json(all_refs)
    print(json_output)
    metadata = extract_metadata_from_pdf("C:/Users/qmerr/Downloads/New_youtube_summarizer/Attention.pdf")
    print(all_refrence_data(metadata, json.loads(json_output)))
    # Optionally save
    with open("references_parsed.json", "w", encoding="utf-8") as f:
        f.write(json_output)