import os
import sys
import json
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client
import gradio as gr
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    place_id: str
    sector_id: str

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings batch: {e}")
        return [[0] * 1536] * len(texts)  # Return zero vectors on error

async def get_title_and_summary(chunk: str, chunk_type: str, area_name: str) -> Dict[str, str]:
    """Extract title and summary using LLM."""

    if chunk_type == "sector_info":
        system_prompt = f"""You are an AI that extracts titles and summaries from climbing sector descriptions.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Extract the sector name. If there are alternative names, include the main one in the title.
        For the summary: Create a concise summary focusing on:
        - Key geographical features
        - Main attractions or characteristics
        - Important facilities (if mentioned)
        - Alternative names for the place (if any)
        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
        Keep both title and summary concise but informative."""
    elif chunk_type == "route":
        system_prompt = f"""You are an AI that extracts titles and summaries from climbing route data.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Extract the route name.
        For the summary: Create a concise summary of the route, including:
        - Grade and difficulty level
        - Type of route (sport, trad, etc.)
        - Notable features or characteristics
        - Length and number of bolts if available
        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
        Keep both title and summary concise but informative."""
    else:  # chunk_type == "route_group" or "routes_collection"
        system_prompt = f"""You are an AI that extracts titles and summaries from collections of climbing routes.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Use 'Rutas de grado {chunk_type.split('_')[-1]} en {area_name}'
        For the summary: Create a concise summary of the routes in this collection, including:
        - Range of difficulty levels
        - Types of routes present
        - Notable features or characteristics
        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
        Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{chunk}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error getting title and summary: {e}")
        return {"title": "Error procesando título", "summary": "Error procesando resumen"}

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "place_id": chunk.place_id,
            "sector_id": chunk.sector_id
        }

        result = supabase.table("rag_data").insert(data).execute()
        logger.info("Inserted chunk")
        return result
    except Exception as e:
        logger.error(f"Error inserting chunk: {e}")
        return None

async def insert_chunks_batch(chunks: List[ProcessedChunk]):
    """Insert multiple chunks in parallel."""
    try:
        data = [
            {
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
                "place_id": chunk.place_id,
                "sector_id": chunk.sector_id
            }
            for chunk in chunks
        ]

        # Supabase upsert - handles both insert and update
        result = supabase.table("rag_data").upsert(data).execute()
        logger.info(f"Inserted {len(chunks)} chunks")
        return result
    except Exception as e:
        logger.error(f"Error inserting chunks: {e}")
        return None

async def delete_existing_chunks(sector_id: str, source: List[str]):
    """Delete existing chunks for this specific sector before inserting new ones."""
    try:
        result = supabase.table("rag_data").delete().eq("sector_id", sector_id).execute()
        logger.info(f"Deleted existing chunks for sector_id: {sector_id}")
        return result
    except Exception as e:
        logger.error(f"Error deleting existing chunks: {e}")
        return None

async def get_sector_data(sector_id: str) -> Dict:
    """Get sector data from Supabase."""
    try:
        response = supabase.table("sector").select("*").eq("id", sector_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error getting sector data: {e}")
        return None

async def get_routes_for_sector(sector_id: str) -> List[Dict]:
    """Get all routes for a sector from Supabase."""
    try:
        response = supabase.table("route").select("*").eq("sector_id", sector_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting routes: {e}")
        return []

async def get_place_id_from_sector(sector_id: str) -> Optional[str]:
    """Get place_id for a sector."""
    try:
        response = supabase.table("sector").select("place_id").eq("id", sector_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]['place_id']
        return None
    except Exception as e:
        logger.error(f"Error getting place_id: {e}")
        return None

async def process_sector_info(sector_data: Dict, place_id: str, metadata_sources: List[str]) -> ProcessedChunk:
    """Process sector information (name, description, approach, ethic)."""
    # Combine sector information into a structured format
    content = f"# {sector_data['name']}\n\n"

    if sector_data.get('description'):
        content += f"## Descripción\n{sector_data['description']}\n\n"

    if sector_data.get('approach'):
        content += f"## Acceso\n{sector_data['approach']}\n\n"

    if sector_data.get('ethic'):
        content += f"## Ética\n{sector_data['ethic']}\n\n"

    # Get title and summary
    extracted = await get_title_and_summary(content, "sector_info", sector_data['name'])

    # Get embedding
    embedding = await get_embedding(content)

    # Create metadata
    metadata = {
        "source": metadata_sources,
        "source_searchable": "|".join(metadata_sources),  # For text search optimization
        "type": "sector_info",
        "chunk_size": len(content),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }

    return ProcessedChunk(
        title=extracted['title'],
        summary=extracted['summary'],
        content=content,
        metadata=metadata,
        embedding=embedding,
        place_id=place_id,
        sector_id=sector_data['id']
    )

async def process_sector_for_rag(sector_data: Dict, routes: List[Dict], place_id: str, metadata_sources: List[str]) -> List[ProcessedChunk]:
    """Process sector and routes with an optimized chunking strategy."""
    chunks = []

    # 1. Sector Overview chunk
    sector_chunk = await process_sector_info(sector_data, place_id, metadata_sources)
    chunks.append(sector_chunk)

    # 2. Group routes by difficulty/grade into logical clusters
    route_groups = defaultdict(list)

    # Group routes by grade prefix (e.g., "5.11", "5.12", etc.)
    for route in routes:
        if not route.get('grade'):
            grade_key = "unknown"
        else:
            # Extract grade prefix (e.g., "5.11" from "5.11c")
            match = re.match(r"(\d+\.\d+)", route.get('grade', ''))
            grade_key = match.group(1) if match else route.get('grade', 'unknown')

        route_groups[grade_key].append(route)

    # Prepare content for each group
    chunk_contents = []
    for grade_key, grouped_routes in route_groups.items():
        # Add sector context
        content = f"# Rutas de grado {grade_key} en {sector_data['name']}\n\n"
        content += f"Información sobre rutas de dificultad {grade_key} en el sector {sector_data['name']}.\n\n"

        # Add routes in this difficulty group
        for route in grouped_routes:
            content += f"## {route['name']}\n\n"

            if route.get('id'):
                content += f"- **ID:** {route['id']}\n"

            if route.get('sector_id'):
                content += f"- **Sector ID:** {route['sector_id']}\n"

            if route.get('grade'):
                content += f"- **Grado:** {route['grade']}\n"

            if route.get('type'):
                content += f"- **Tipo:** {route['type']}\n"

            if route.get('quality'):
                content += f"- **Calidad:** {route['quality']}\n"

            if route.get('length'):
                content += f"- **Longitud:** {route['length']}\n"

            if route.get('bolts'):
                content += f"- **Bolts:** {route['bolts']}\n"

            if route.get('description'):
                content += f"\n{route['description']}\n"

            content += "\n"

        chunk_contents.append({
            "content": content,
            "grade_key": grade_key,
            "route_count": len(grouped_routes)
        })

    # Get embeddings in batch
    texts = [c["content"] for c in chunk_contents]
    embeddings = await get_embeddings_batch(texts)

    # Get title and summary for each chunk and create ProcessedChunk objects
    processed_group_chunks = []
    for i, chunk_data in enumerate(chunk_contents):
        extracted = await get_title_and_summary(chunk_data["content"], f"route_group_{chunk_data['grade_key']}", sector_data['name'])

        metadata = {
            "source": metadata_sources,
            "source_searchable": "|".join(metadata_sources),  # For text search optimization
            "type": "route_group",
            "grade_group": chunk_data["grade_key"],
            "sector_name": sector_data['name'],
            "route_count": chunk_data["route_count"],
            "chunk_size": len(chunk_data["content"]),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        }

        processed_group_chunks.append(ProcessedChunk(
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk_data["content"],
            metadata=metadata,
            embedding=embeddings[i],
            place_id=place_id,
            sector_id=sector_data['id']
        ))

    # Add route group chunks to the chunks list
    chunks.extend(processed_group_chunks)

    return chunks

async def generate_chunks_preview(sector_id: str, metadata_sources: List[str]):
    """Generate chunks for preview without inserting them into the database."""
    # Get sector data
    sector_data = await get_sector_data(sector_id)
    if not sector_data:
        logger.error(f"Could not find sector with id: {sector_id}")
        return None, f"Error: Could not find sector with id: {sector_id}"

    # Get place_id
    place_id = await get_place_id_from_sector(sector_id)
    if not place_id:
        logger.error(f"No place_id found for sector: {sector_id}")
        return None, f"Error: No place_id found for sector: {sector_id}"

    # Get routes for this sector
    routes = await get_routes_for_sector(sector_id)

    # Process sector for RAG using the optimized chunking strategy
    chunks = await process_sector_for_rag(sector_data, routes, place_id, metadata_sources)

    return chunks, f"Generated {len(chunks)} chunks for preview. Review them before inserting to the database."

async def insert_chunks_to_db(chunks: List[ProcessedChunk], place_id: str, metadata_sources: List[str]):
    """Insert the previewed chunks into the database."""
    if not chunks:
        return "No chunks to insert."

    # Delete existing chunks for this sector_id
    sector_id = chunks[0].sector_id
    await delete_existing_chunks(sector_id, metadata_sources)

    # Insert all chunks in batch
    result = await insert_chunks_batch(chunks)

    return f"Successfully inserted {len(chunks)} chunks into the database."

async def process_sector_and_routes(sector_id: str, metadata_sources: List[str]):
    """Process a climbing sector and all its routes."""
    # Get sector data
    sector_data = await get_sector_data(sector_id)
    if not sector_data:
        logger.error(f"Could not find sector with id: {sector_id}")
        return f"Error: Could not find sector with id: {sector_id}"

    # Get place_id
    place_id = await get_place_id_from_sector(sector_id)
    if not place_id:
        logger.error(f"No place_id found for sector: {sector_id}")
        return f"Error: No place_id found for sector: {sector_id}"

    # Delete existing chunks for this sector_id
    await delete_existing_chunks(sector_id, metadata_sources)

    # Get routes for this sector
    routes = await get_routes_for_sector(sector_id)

    # Process sector for RAG using the optimized chunking strategy
    chunks = await process_sector_for_rag(sector_data, routes, place_id, metadata_sources)

    # Insert all chunks in batch
    await insert_chunks_batch(chunks)

    return f"Successfully processed sector {sector_data['name']} with {len(routes)} routes into {len(chunks)} optimized chunks"

async def get_all_sectors():
    """Get list of all sectors for UI dropdown."""
    try:
        response = supabase.table("sector").select("id, name").order("name").execute()
        logger.info(f"Retrieved {len(response.data)} sectors from database")
        return response.data
    except Exception as e:
        logger.error(f"Error getting sectors: {e}")
        return []

async def get_sector_details(sector_id: str) -> Dict:
    """Get detailed information about a sector including climbing type."""
    try:
        response = supabase.table("sector").select("*").eq("id", sector_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error getting sector details: {e}")
        return None

def create_ui():
    """Create Gradio UI for sector selection and processing."""
    # Define metadata source options
    metadata_options = [
        "candelas",
        "salitre",
        "panales",
        "san cayetano",
        "zelda",
        "comadres",
        "guadalcazar"
    ]

    # Create a list to store generated chunks for preview
    generated_chunks = []
    current_sector_id = None

    async def on_generate_preview(sector_id, metadata_sources):
        # Process selected sector with selected metadata sources to generate preview
        if not metadata_sources:
            return "Error: Please select at least one metadata source", None, [], False

        nonlocal generated_chunks
        nonlocal current_sector_id

        chunks, message = await generate_chunks_preview(sector_id, metadata_sources)

        if chunks:
            generated_chunks = chunks
            # Get sector_id for later use during insertion
            current_sector_id = sector_id

            # Prepare data for the preview table
            preview_data = []
            for i, chunk in enumerate(chunks):
                preview_data.append([
                    i+1,
                    chunk.title,
                    chunk.summary,
                    chunk.metadata.get('type', 'Unknown'),
                    len(chunk.content)
                ])

            return message, gr.update(visible=True), preview_data, True
        else:
            return message, gr.update(visible=False), [], False

    async def on_insert_to_db(metadata_sources):
        # Insert the previewed chunks into the database
        nonlocal generated_chunks
        nonlocal current_sector_id

        if not generated_chunks or not current_sector_id:
            return "No chunks to insert. Please generate a preview first."

        # Get place_id for UI display purposes
        place_id = generated_chunks[0].place_id if generated_chunks else None

        result = await insert_chunks_to_db(generated_chunks, place_id, metadata_sources)

        # Clear the chunks after insertion
        generated_chunks = []

        return result

    async def update_sector_dropdown():
        sectors = await get_all_sectors()
        choices = [(s['name'], s['id']) for s in sectors]
        logger.info(f"Updating dropdown with {len(choices)} sector choices")
        return choices

    async def update_climbing_type(sector_id):
        """Update the climbing type tag when a sector is selected."""
        if not sector_id:
            return "N/A"

        sector_data = await get_sector_details(sector_id)
        if not sector_data or not sector_data.get('climbing_type'):
            return "Tipo: No especificado"

        # climbing_type is stored as a JSON array
        try:
            climbing_types = sector_data['climbing_type']
            if isinstance(climbing_types, str):
                climbing_types = json.loads(climbing_types)

            if not climbing_types:
                return "Tipo: No especificado"

            types_str = ", ".join(climbing_types)
            return f"Tipo: {types_str}"
        except Exception as e:
            logger.error(f"Error parsing climbing type: {e}")
            return "Tipo: Error"

    # Get sectors synchronously for initial dropdown population
    def get_initial_sectors():
        response = supabase.table("sector").select("id, name").order("name").execute()
        choices = [(s['name'], s['id']) for s in response.data]
        logger.info(f"Initial dropdown populated with {len(choices)} sector choices")
        return choices

    initial_sectors = get_initial_sectors()

    with gr.Blocks(title="RAG Data Update Tool") as app:
        gr.Markdown("# RAG Data Update Tool")
        gr.Markdown("Select a climbing sector to process for RAG data")

        # Load sectors on startup with pre-fetched data
        sector_dropdown = gr.Dropdown(
            label="Select Sector",
            choices=initial_sectors,
            info="Choose a climbing sector to process",
            scale=1
        )

        # Add climbing type tag
        climbing_type_tag = gr.Textbox(
            label="Climbing Type",
            value="Tipo: No seleccionado",
            interactive=False
        )

        # Update climbing type when sector selection changes
        sector_dropdown.change(
            fn=update_climbing_type,
            inputs=[sector_dropdown],
            outputs=[climbing_type_tag]
        )

        refresh_btn = gr.Button("Refresh Sector List")
        refresh_btn.click(fn=update_sector_dropdown, outputs=[sector_dropdown])

        # Replace text input with multiselect dropdown
        metadata_sources = gr.Dropdown(
            label="Metadata Sources",
            choices=metadata_options,
            multiselect=True,
            value=["san cayetano", "guadalcazar"],
            info="Select metadata sources for the chunks"
        )

        # Preview button instead of direct processing
        preview_btn = gr.Button("Generate Preview")
        preview_result = gr.Textbox(label="Preview Result", interactive=False)

        # Preview table container that is initially hidden
        with gr.Column(visible=False) as preview_container:
            gr.Markdown("### Generated Chunks Preview")
            preview_table = gr.Dataframe(
                headers=["#", "Title", "Summary", "Type", "Content Length"],
                datatype=["number", "str", "str", "str", "number"],
                label="Chunks Preview"
            )

            # Insert button only visible after preview is generated
            insert_btn = gr.Button("Insert to Database", variant="primary")
            insert_result = gr.Textbox(label="Insert Result", interactive=False)

        # Connect preview button to generate preview function
        preview_btn.click(
            fn=on_generate_preview,
            inputs=[sector_dropdown, metadata_sources],
            outputs=[preview_result, preview_container, preview_table, insert_btn]
        )

        # Connect insert button to insert function
        insert_btn.click(
            fn=on_insert_to_db,
            inputs=[metadata_sources],
            outputs=[insert_result]
        )

    return app

async def main():
    app = create_ui()
    app.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    # Run the async app inside asyncio event loop
    asyncio.run(main())
