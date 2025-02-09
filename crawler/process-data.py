import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Get place_id for Guadalcazar
result = supabase.table("Place").select("id").eq("name", "Guadalcazar").execute()
place_id = result.data[0]["id"] if result.data else None

if not place_id:
    raise ValueError("Place 'Guadalcazar' not found in the database")

@dataclass
class ProcessedChunk:
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    place_id: str

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def get_title_and_summary(chunk: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from data chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: Content:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "place_id": chunk.place_id
        }

        result = supabase.table("place_data").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_chunk(chunk: str, chunk_number: int) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": "cactux",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }

    return ProcessedChunk(
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding,
        place_id=place_id
    )

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, keeping metadata sections together and respecting route boundaries."""
    chunks = []

    # Find the end of the metadata section (Title through Ethics)
    metadata_sections = ["# ", "## Description", "## Approach", "## Ethics"]
    last_metadata_pos = -1

    for section in metadata_sections:
        pos = text.find(section)
        if pos != -1:
            section_end = text.find("\n## ", pos + len(section))
            if section_end != -1:
                last_metadata_pos = max(last_metadata_pos, section_end)

    # If we found metadata sections, make them the first chunk
    if last_metadata_pos != -1:
        first_chunk = text[:last_metadata_pos].strip()
        if first_chunk:
            chunks.append(first_chunk)
        remaining_text = text[last_metadata_pos:]
    else:
        remaining_text = text

    # Process the rest of the text
    start = 0
    text_length = len(remaining_text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(remaining_text[start:].strip())
            break

        # Try to find a route boundary (### )
        chunk = remaining_text[start:end]
        next_route = chunk.rfind('\n### ')
        if next_route != -1 and next_route > chunk_size * 0.3:
            end = start + next_route

        # If no route boundary, try to break at a paragraph
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # Extract chunk and clean it up
        chunk = remaining_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def read_guadalcazar_md(filename):
    # Define the directory path
    base_path = "./climbs/san_luis_potosi/guadalcazar"
    file_path = os.path.join(base_path, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

async def main():
    # Specify the filename you want to read
    filename = "guadalcazar.md"  # Change this to your desired filename
    content = read_guadalcazar_md(filename)
    if content:
        print("Processing file contents...")
        chunks = chunk_text(content)

        # Process chunks in parallel
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        processed_chunks = await asyncio.gather(*tasks)

         # Store chunks in parallel
        insert_tasks = [
            insert_chunk(chunk)
            for chunk in processed_chunks
        ]
        await asyncio.gather(*insert_tasks)

if __name__ == "__main__":
    asyncio.run(main())
