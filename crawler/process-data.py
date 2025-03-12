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

area_name = "Gruta de las Candelas"

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Get place_id for Guadalcazar - using title case for the area name
result = supabase.table("place").select("id").eq("name", "Guadalcazar").execute()
place_id = result.data[0]["id"] if result.data else None

if not place_id:
    raise ValueError(f"Place Guadalcazar not found in the database")

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

async def get_title_and_summary(chunk: str, is_place: bool = False) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    if is_place:
        system_prompt = f"""You are an AI that extracts titles and summaries from place descriptions.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Extract the place name. If there are alternative names, include the main one in the title.
        For the summary: Create a concise summary focusing on:
        - Key geographical features
        - Main attractions or characteristics
        - Important facilities (if mentioned)
        - Alternative names for the place (if any)
        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
        Keep both title and summary concise but informative."""
    else:
        system_prompt = f"""You are an AI that extracts titles and summaries from data chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, it will be route information, so make sure 'Rutas de escalada en {area_name} (Zona)' is the title.
        For the summary: Create a concise summary of the main points in this chunk, include alternative names for the area if there are any.
        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
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

        result = supabase.table("rag_data").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_chunk(chunk: str, chunk_number: int) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, is_place=False)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": ["candelas", "guadalcazar"], #IMPORTANT: Add the source of the data for each different source
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
    """Split text into chunks, with metadata sections together and routes in separate chunks."""
    chunks = []

    # First chunk: Title through Ethics sections
    metadata_sections = ["# ", "## Descripción", "## Acceso", "## Ética"]
    routes_start = text.find("## Rutas")

    if routes_start != -1:
        # Add metadata sections as first chunk
        metadata_chunk = text[:routes_start].strip()
        if metadata_chunk:
            chunks.append(metadata_chunk)

        # Process routes section
        routes_text = text[routes_start:]
        current_chunk = []
        current_size = 0

        # Split routes into lines
        for line in routes_text.split('\n'):
            # Start new route section
            if line.startswith('### '):
                # If we have content and exceed chunk size, save current chunk
                if current_size >= chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline

        # Add remaining chunk if any
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

    return chunks

def chunk_text_places(text: str, chunk_size: int = 5000) -> List[str]:
    """Split place description text into logical chunks based on main sections.
    First chunk contains core info (name, description, access, ethics),
    subsequent chunks contain auxiliary information (lodging, food, etc.)."""
    chunks = []

    # Define main section markers
    core_sections = ["# ", "## Descripción", "## Acceso", "## Ética"]
    auxiliary_sections = ["## Hospedaje", "## Alimentos", "## Transporte", "## Notas"]

    # Find the start of the first auxiliary section
    aux_start = -1
    for section in auxiliary_sections:
        pos = text.find(section)
        if pos != -1 and (aux_start == -1 or pos < aux_start):
            aux_start = pos

    if aux_start != -1:
        # First chunk: Core information (name through ethics)
        core_chunk = text[:aux_start].strip()
        if core_chunk:
            chunks.append(core_chunk)

        # Process auxiliary sections
        current_chunk = []
        current_size = 0

        # Split remaining text into lines
        for line in text[aux_start:].split('\n'):
            # Start new section
            if any(line.startswith(section) for section in auxiliary_sections):
                # If we have content and exceed chunk size, save current chunk
                if current_size >= chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline

        # Add remaining chunk if any
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
    else:
        # If no auxiliary sections found, return entire text as one chunk
        chunks.append(text)

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
    filename = f"{area_name}.md"  # Change this to your desired filename
    content = read_guadalcazar_md(filename)

    if content:
        print("Processing file contents...")
        chunks = chunk_text(content)

        # Process chunks in parallel
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        processed_chunks = await asyncio.gather(*tasks)

       # Print the processed chunks
        # for chunk in processed_chunks:
        #     print("\n" + "="*80)  # Clear separator line
        #     print(f"CHUNK {chunk.chunk_number}")
        #     print("="*80)
        #     print(f"TITLE: {chunk.title}")
        #     print(f"SUMMARY: {chunk.summary}")
        #     print("\nCONTENT:")
        #     print(chunk.content)
        #     print("\nMETADATA:")
        #     print(chunk.metadata)
        #     print("="*80 + "\n")  # Bottom separator line

         # Store chunks in parallel
        insert_tasks = [
            insert_chunk(chunk)
            for chunk in processed_chunks
        ]
        await asyncio.gather(*insert_tasks)

if __name__ == "__main__":
    asyncio.run(main())
