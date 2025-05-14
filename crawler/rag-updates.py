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
from google.generativeai import GenerativeModel
import google.generativeai as genai
from supabase import create_client, Client
import gradio as gr
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Google Generative AI client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialize Supabase client
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
    place_id: Optional[str] = None
    sector_id: Optional[str] = None
    business_id: Optional[str] = None

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Google."""
    try:
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            output_dimensionality=768,
            task_type="SEMANTIC_SIMILARITY"
        )
        return embedding["embedding"]
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 768  # Return zero vector on error (Google embeddings are 768 dimensions)

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call."""
    try:
        # Currently, we need to call the embedding API for each text separately
        # as the batch API might have limitations
        embeddings = []
        for text in texts:
            embedding = await get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings batch: {e}")
        return [[0] * 768] * len(texts)  # Return zero vectors on error

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
    elif chunk_type.startswith("boulder_group"):
        # Extract the grade from the chunk_type
        grade_key = chunk_type.split('_')[-1]

        # Set specific title and special prompts for specific grades
        if grade_key == "desconocido":
            title = f"Bloques de {area_name} - Grado Desconocido"
            system_prompt = f"""You are an AI that extracts summaries from boulder climbing data.
            Return a JSON object with 'title' and 'summary' keys.
            For the title: Use '{title}'
            For the summary: Create a concise summary of the boulder problem(s), emphasizing that:
            - These boulders have unknown grades that need to be determined
            - Climbers are invited to try these problems and help establish their grades
            - Include any style characteristics (sloper, crimpy, etc.) if mentioned
            - Mention any notable features or special conditions

            All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
            Keep the summary concise but informative, and be sure to include the invitation for climbers to determine the grades."""
        elif grade_key == "proyecto-abierto":
            title = f"Bloques de {area_name} - Proyectos Abiertos"
            system_prompt = f"""You are an AI that extracts summaries from boulder climbing data.
            Return a JSON object with 'title' and 'summary' keys.
            For the title: Use '{title}'
            For the summary: Create a concise summary of the boulder problem(s), emphasizing that:
            - These are new open projects that haven't been completed yet
            - Climbers are invited to attempt these problems and potentially make first ascents
            - Include any style characteristics (sloper, crimpy, etc.) if mentioned
            - Mention any notable features or special conditions

            All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
            Keep the summary concise but informative, and be sure to include the invitation for climbers to try these open projects."""
        else:
            title = f"Bloques de {area_name} - Grado {grade_key}"
            system_prompt = f"""You are an AI that extracts summaries from boulder climbing data.
            Return a JSON object with 'title' and 'summary' keys.
            For the title: Use '{title}'
            For the summary: Create a concise summary of the boulder problem(s), including:
            - Grade and difficulty level
            - Style characteristics (sloper, crimpy, etc.)
            - Notable features like "top quality" or "highball"
            - Special conditions (sun/shade, approach, etc.)

            All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
            Keep the summary concise but informative."""
    elif chunk_type == "business_info":
        system_prompt = f"""You are an AI that extracts titles and summaries from business information.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Use the business name and add a descriptive phrase about its type or main service.
        For the summary: Create a concise summary focusing on:
        - What type of business this is (accommodation, restaurant, market, gas station, etc.)
        - Main services or products offered
        - Key features or amenities (WiFi, delivery, etc.)
        - Location or relevance to visitors/climbers if mentioned

        Examples of good titles:
        - "Hotel Montaña - Alojamiento cercano a zonas de escalada"
        - "Restaurante El Peñón - Comida local para escaladores"
        - "Mercado San Cayetano - Provisiones para tu aventura"
        - "Gasolinera Roca - Combustible y servicios básicos"

        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
        Keep both title and summary concise but informative."""
    elif chunk_type == "business_menu":
        system_prompt = f"""You are an AI that extracts titles and summaries from restaurant menus.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: Use "Menú de [Business Name]" with a descriptive subtitle about the cuisine type.
        For the summary: Create a concise summary focusing on:
        - Type of cuisine or food style
        - Highlight of popular or specialty dishes
        - Price range or value proposition if mentioned
        - Special dietary options (vegetarian, vegan, etc.) if available

        Examples of good titles:
        - "Menú de La Cumbre - Cocina tradicional de montaña"
        - "Menú de Café Escalada - Opciones rápidas para escaladores"

        All data is in Spanish. IMPORTANT: Return the title and summary in Spanish.
        Keep both title and summary concise but informative."""
    else:  # chunk_type.startswith("route_group")
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
        # Create Gemini model
        model = GenerativeModel("gemini-2.5-flash-preview-04-17")

        # Define prompt for Gemini - it only supports 'user' and 'model' roles
        prompt = [
            # Gemini doesn't support system role, include system instructions as user message
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\nContent:\n{chunk}"}]}
        ]

        # Make the request to Gemini
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        # Parse the JSON response
        result = json.loads(response.text)

        # For boulder groups, we want to ensure the title follows our format
        if chunk_type.startswith("boulder_group"):
            grade_key = chunk_type.split('_')[-1]
            if grade_key == "desconocido":
                result["title"] = f"Bloques de {area_name} - Grado Desconocido"
            elif grade_key == "proyecto-abierto":
                result["title"] = f"Bloques de {area_name} - Proyectos Abiertos"
            else:
                result["title"] = f"Bloques de {area_name} - Grado {grade_key}"

        return result
    except Exception as e:
        logger.error(f"Error getting title and summary: {e}")

        # If there's an error but we have a boulder group, return a default title
        if chunk_type.startswith("boulder_group"):
            grade_key = chunk_type.split('_')[-1]
            if grade_key == "desconocido":
                title = f"Bloques de {area_name} - Grado Desconocido"
                summary = "Bloques con grado por determinar. Se invita a los escaladores a intentar estos problemas y ayudar a establecer sus grados."
            elif grade_key == "proyecto-abierto":
                title = f"Bloques de {area_name} - Proyectos Abiertos"
                summary = "Proyectos abiertos que aún no han sido completados. Se invita a los escaladores a intentar estos problemas y potencialmente hacer primeras ascensiones."
            else:
                title = f"Bloques de {area_name} - Grado {grade_key}"
                summary = f"Información sobre bloques de dificultad {grade_key} en el sector {area_name}."

            return {"title": title, "summary": summary}

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
            "sector_id": chunk.sector_id,
            "business_id": chunk.business_id
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
                "sector_id": chunk.sector_id,
                "business_id": chunk.business_id
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
        # Only delete records where sector_id matches the specified sector_id
        # This ensures we don't delete records that might not have a sector_id at all
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

async def get_boulders_for_sector(sector_id: str) -> List[Dict]:
    """Get all boulders for a sector from Supabase."""
    try:
        response = supabase.table("boulder").select("*").eq("sector_id", sector_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting boulders: {e}")
        return []

def translate_boulder_style(style_list):
    """Translate boulder style options from English to Spanish."""
    # Map for translating style options
    style_map = {
        "Flat approach": "Aproximación plana",
        "Uphill approach": "Aproximación en subida",
        "Steep uphill approach": "Aproximación en subida pronunciada",
        "Downhill approach": "Aproximación en bajada",
        "Morning sun": "Sol de mañana",
        "Afternoon sun": "Sol de tarde",
        "Tree-filtered sun (am)": "Sol filtrado por árboles (mañana)",
        "Tree-filtered sun (pm)": "Sol filtrado por árboles (tarde)",
        "Sunny most of the day": "Soleado la mayor parte del día",
        "Shady most of the day": "Sombreado la mayor parte del día",
        "Boulders dry fast": "Los bloques se secan rápido",
        "Boulders dry in rain": "Los bloques se escalan bajo la lluvia",
        "Start seated": "Inicio sentado",
        '"Highball", dangerous': '"Highball", peligroso',
        "Slabby problem": "Problema de Slab",
        "Very steep problem": "Problema muy desplomado",
        "Reachy, best if tall": "Morfo, mejor si eres alto",
        "Dynamic": "Dinámico",
        "Pumpy or sustained": "Bombeador o sostenido",
        "Technical": "Técnico",
        "Powerful": "Potente",
        "Pockets": "Pockets",
        "Small edges, crimpy": "Regletas, crimpy",
        "Slopey holds": "Agarres de Sloper"
    }

    translated = []
    for style in style_list:
        if style in style_map:
            translated.append(style_map[style])
        else:
            translated.append(style)  # Keep original if no translation found

    return translated

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

async def process_sector_info(sector_data: Dict, place_id: str, metadata_sources: List[str], is_boulder_sector: bool = False) -> ProcessedChunk:
    """Process sector information (name, description, approach, ethic)."""
    # Combine sector information into a structured format
    content = f"# {sector_data['name']}\n\n"

    if sector_data.get('description'):
        content += f"## Descripción\n{sector_data['description']}\n\n"

    if sector_data.get('approach'):
        content += f"## Acceso\n{sector_data['approach']}\n\n"

    if sector_data.get('ethic'):
        content += f"## Ética\n{sector_data['ethic']}\n\n"

    # For boulder sectors, add a clear indication
    if is_boulder_sector:
        content += f"## Tipo de Escalada\nEste es un sector de boulder.\n\n"

    # Get title and summary
    extracted = await get_title_and_summary(content, "sector_info", sector_data['name'])

    # Get embedding
    embedding = await get_embedding(content)

    # Create metadata
    metadata = {
        "source": metadata_sources,
        "source_searchable": "|".join(metadata_sources),  # For text search optimization
        "type": "sector_info",
        "sector_type": "boulder" if is_boulder_sector else "route",
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

    # Check if this is a boulder sector
    is_boulder_sector = False
    if sector_data.get('climbing_type'):
        # Parse climbing_type if it's a string (JSON array)
        climbing_types = sector_data['climbing_type']
        if isinstance(climbing_types, str):
            try:
                climbing_types = json.loads(climbing_types)
            except Exception as e:
                logger.error(f"Error parsing climbing_type: {e}")
                climbing_types = []

        if climbing_types == ["boulder"]:
            is_boulder_sector = True

    # 1. Sector Overview chunk - now passing the is_boulder_sector flag
    sector_chunk = await process_sector_info(sector_data, place_id, metadata_sources, is_boulder_sector)
    chunks.append(sector_chunk)

    # 2. Group routes by difficulty/grade into logical clusters
    route_groups = defaultdict(list)

    # Group routes by grade prefix
    for route in routes:
        if not route.get('grade'):
            grade_key = "unknown"
        else:
            grade = route.get('grade', '')

            # Handle special boulder grades
            if is_boulder_sector:
                if grade == "?":
                    grade_key = "desconocido"
                    # Update the grade for display purposes
                    route['grade_display'] = "Desconocido"
                elif grade == "P.A.":
                    grade_key = "proyecto-abierto"
                    # Update the grade for display purposes
                    route['grade_display'] = "Proyecto Abierto"
                else:
                    # For boulder grades like V0, V1, etc., use the whole grade as the key
                    grade_key = grade
                    route['grade_display'] = grade
            else:
                # For routes, extract grade prefix (e.g., "5.11" from "5.11c")
                match = re.match(r"(\d+\.\d+)", grade)
                grade_key = match.group(1) if match else grade
                route['grade_display'] = grade

        route_groups[grade_key].append(route)

    # Prepare content for each group
    chunk_contents = []
    for grade_key, grouped_routes in route_groups.items():
        # Add sector context
        if is_boulder_sector:
            # Properly display the grade name in the title
            if grade_key == "desconocido":
                display_grade = "Desconocido"
            elif grade_key == "proyecto-abierto":
                display_grade = "Proyecto Abierto"
            else:
                display_grade = grade_key

            content = f"# Bloques de grado {display_grade} en {sector_data['name']}\n\n"
            content += f"Información sobre bloques de dificultad {display_grade} en el sector {sector_data['name']}.\n\n"
        else:
            content = f"# Rutas de grado {grade_key} en {sector_data['name']}\n\n"
            content += f"Información sobre rutas de dificultad {grade_key} en el sector {sector_data['name']}.\n\n"

        # Add routes in this difficulty group
        for route in grouped_routes:
            content += f"## {route['name']}\n\n"

            if route.get('id'):
                content += f"- **ID:** {route['id']}\n"

            if route.get('sector_id'):
                content += f"- **Sector ID:** {route['sector_id']}\n"

            # Use the display grade if available
            if route.get('grade_display'):
                content += f"- **Grado:** {route['grade_display']}\n"
            elif route.get('grade'):
                content += f"- **Grado:** {route['grade']}\n"

            if route.get('type'):
                content += f"- **Tipo:** {route['type']}\n"

            if route.get('quality'):
                content += f"- **Calidad:** {route['quality']}\n"

            # Add boulder-specific fields
            if is_boulder_sector:
                if route.get('top') is not None:
                    content += f"- **TOP:** {'Sí' if route['top'] else 'No'}\n"

                if route.get('style'):
                    try:
                        # Parse the style JSON data
                        style_data = route['style']
                        if isinstance(style_data, str):
                            style_data = json.loads(style_data)

                        # Translate style options
                        translated_styles = translate_boulder_style(style_data)

                        if translated_styles:
                            content += f"- **Características:** {', '.join(translated_styles)}\n"
                    except Exception as e:
                        logger.error(f"Error parsing boulder style: {e}")

            if route.get('length'):
                content += f"- **Longitud:** {route['length']}\n"

            if route.get('bolts') and not is_boulder_sector:
                content += f"- **Bolts:** {route['bolts']}\n"

            if route.get('height') and is_boulder_sector:
                content += f"- **Altura:** {route['height']}\n"

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
        chunk_type = "boulder_group" if is_boulder_sector else "route_group"
        extracted = await get_title_and_summary(chunk_data["content"], f"{chunk_type}_{chunk_data['grade_key']}", sector_data['name'])

        metadata = {
            "source": metadata_sources,
            "source_searchable": "|".join(metadata_sources),  # For text search optimization
            "type": chunk_type,
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

    # Check if this is a boulder sector
    is_boulder_sector = False
    if sector_data.get('climbing_type'):
        # Parse climbing_type if it's a string (JSON array)
        climbing_types = sector_data['climbing_type']
        if isinstance(climbing_types, str):
            try:
                climbing_types = json.loads(climbing_types)
            except Exception as e:
                logger.error(f"Error parsing climbing_type: {e}")
                climbing_types = []

        if climbing_types == ["boulder"]:
            is_boulder_sector = True

    # Get routes or boulders for this sector
    if is_boulder_sector:
        routes = await get_boulders_for_sector(sector_id)
        logger.info(f"Retrieved {len(routes)} boulders for sector: {sector_id}")
    else:
        routes = await get_routes_for_sector(sector_id)
        logger.info(f"Retrieved {len(routes)} routes for sector: {sector_id}")

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

    # Check if this is a boulder sector
    is_boulder_sector = False
    if sector_data.get('climbing_type'):
        # Parse climbing_type if it's a string (JSON array)
        climbing_types = sector_data['climbing_type']
        if isinstance(climbing_types, str):
            try:
                climbing_types = json.loads(climbing_types)
            except Exception as e:
                logger.error(f"Error parsing climbing_type: {e}")
                climbing_types = []

        if climbing_types == ["boulder"]:
            is_boulder_sector = True

    # Get routes or boulders for this sector
    if is_boulder_sector:
        routes = await get_boulders_for_sector(sector_id)
        logger.info(f"Processing sector {sector_data['name']} with {len(routes)} boulders")
    else:
        routes = await get_routes_for_sector(sector_id)
        logger.info(f"Processing sector {sector_data['name']} with {len(routes)} routes")

    # Process sector for RAG using the optimized chunking strategy
    chunks = await process_sector_for_rag(sector_data, routes, place_id, metadata_sources)

    # Insert all chunks in batch
    await insert_chunks_batch(chunks)

    entity_type = "boulders" if is_boulder_sector else "routes"
    return f"Successfully processed sector {sector_data['name']} with {len(routes)} {entity_type} into {len(chunks)} optimized chunks"

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

async def get_all_businesses():
    """Get list of all businesses for UI dropdown."""
    try:
        response = supabase.table("business").select("id, name").order("name").execute()
        logger.info(f"Retrieved {len(response.data)} businesses from database")
        return response.data
    except Exception as e:
        logger.error(f"Error getting businesses: {e}")
        return []

async def get_business_data(business_id: str) -> Dict:
    """Get business data from Supabase."""
    try:
        response = supabase.table("business").select("*").eq("id", business_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error getting business data: {e}")
        return None

async def delete_existing_business_chunks(business_id: str, source: List[str]):
    """Delete existing chunks for this specific business before inserting new ones."""
    try:
        # Only delete records where business_id matches the specified business_id
        # This ensures we don't delete records that might not have a business_id at all
        result = supabase.table("rag_data").delete().eq("business_id", business_id).execute()
        logger.info(f"Deleted existing chunks for business_id: {business_id}")
        return result
    except Exception as e:
        logger.error(f"Error deleting existing business chunks: {e}")
        return None

async def process_business_info(business_data: Dict, metadata_sources: List[str]) -> ProcessedChunk:
    """Process business information into a structured chunk."""
    # Combine business information into a structured format
    content = f"# {business_data['name']}\n\n"

    # Get business type for better context
    business_type_list = []
    if business_data.get('type'):
        # Handle type if it's a JSON array
        business_types = business_data['type']
        if isinstance(business_types, str):
            try:
                business_types = json.loads(business_types)
                business_type_list = business_types
            except Exception as e:
                logger.error(f"Error parsing business type: {e}")
                business_types = []
        else:
            business_type_list = business_types

        if business_types:
            content += f"## Tipo de Negocio\n{', '.join(business_types)}\n\n"

    if business_data.get('description'):
        content += f"## Descripción\n{business_data['description']}\n\n"

    if business_data.get('phone'):
        content += f"## Teléfono\n{business_data['phone']}\n\n"

    if business_data.get('hours'):
        hours_data = business_data['hours']
        if isinstance(hours_data, str):
            try:
                hours_data = json.loads(hours_data)
            except Exception as e:
                logger.error(f"Error parsing business hours: {e}")
                hours_data = {}

        if hours_data:
            content += "## Horario\n"
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            day_names = {
                "monday": "Lunes",
                "tuesday": "Martes",
                "wednesday": "Miércoles",
                "thursday": "Jueves",
                "friday": "Viernes",
                "saturday": "Sábado",
                "sunday": "Domingo"
            }

            for day in days:
                if day in hours_data and hours_data[day]:
                    day_schedule = hours_data[day]
                    if isinstance(day_schedule, list) and day_schedule:
                        slots = []
                        for slot in day_schedule:
                            if 'open' in slot and 'close' in slot:
                                slots.append(f"{slot['open']} - {slot['close']}")
                        if slots:
                            content += f"- {day_names[day]}: {', '.join(slots)}\n"
            content += "\n"

    # Add delivery and wifi information
    features = []
    if business_data.get('has_delivery') is True:
        features.append("Servicio a domicilio disponible")
    if business_data.get('has_free_wifi') is True:
        features.append("WiFi gratuito disponible")

    if features:
        content += f"## Características\n"
        for feature in features:
            content += f"- {feature}\n"
        content += "\n"

    # Add social media information
    social_media = []
    if business_data.get('instagram'):
        social_media.append(f"Instagram: {business_data['instagram']}")
    if business_data.get('facebook'):
        social_media.append(f"Facebook: {business_data['facebook']}")

    if social_media:
        content += f"## Redes Sociales\n"
        for social in social_media:
            content += f"- {social}\n"
        content += "\n"

    # Process menu if available
    if business_data.get('menu'):
        menu_data = business_data['menu']
        if isinstance(menu_data, str):
            try:
                menu_data = json.loads(menu_data)
            except Exception as e:
                logger.error(f"Error parsing menu data: {e}")
                menu_data = {}

        if menu_data and isinstance(menu_data, list):
            content += "## Menú\n"
            for category in menu_data:
                if 'name' in category:
                    content += f"### {category['name']}\n"
                    if 'items' in category and isinstance(category['items'], list):
                        for item in category['items']:
                            item_info = []
                            if 'name' in item:
                                item_info.append(f"**{item['name']}**")
                            if 'price' in item:
                                item_info.append(f"${item['price']}")
                            if 'description' in item:
                                item_info.append(f"{item['description']}")

                            if item_info:
                                content += f"- {' - '.join(item_info)}\n"
                    content += "\n"

    # Get title and summary using enhanced prompts for businesses
    extracted = await get_title_and_summary(content, "business_info", business_data['name'])

    # Get embedding
    embedding = await get_embedding(content)

    # Create metadata
    metadata = {
        "source": metadata_sources,
        "source_searchable": "|".join(metadata_sources),  # For text search optimization
        "type": "business_info",
        "business_type": business_type_list,
        "chunk_size": len(content),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }

    return ProcessedChunk(
        title=extracted['title'],
        summary=extracted['summary'],
        content=content,
        metadata=metadata,
        embedding=embedding,
        place_id=None,  # Businesses don't have a place_id
        sector_id=None,  # Businesses don't have a sector_id
        business_id=business_data['id']  # Add business_id
    )

async def process_business_menu(business_data: Dict, metadata_sources: List[str]) -> Optional[ProcessedChunk]:
    """Process business menu into a separate chunk if menu exists and is substantial."""
    if not business_data.get('menu'):
        return None

    menu_data = business_data['menu']
    if isinstance(menu_data, str):
        try:
            menu_data = json.loads(menu_data)
        except Exception as e:
            logger.error(f"Error parsing menu data: {e}")
            return None

    if not menu_data or not isinstance(menu_data, list) or len(menu_data) == 0:
        return None

    # Create content specifically for the menu
    content = f"# Menú de {business_data['name']}\n\n"

    for category in menu_data:
        if 'name' in category:
            content += f"## {category['name']}\n"
            if 'items' in category and isinstance(category['items'], list):
                for item in category['items']:
                    item_info = []
                    if 'name' in item:
                        item_info.append(f"**{item['name']}**")
                    if 'price' in item:
                        item_info.append(f"${item['price']}")
                    if 'description' in item:
                        item_info.append(f"{item['description']}")

                    if item_info:
                        content += f"- {' - '.join(item_info)}\n"
            content += "\n"

    # Only create a menu chunk if there's substantial content
    if len(content) < 100:  # Arbitrary threshold
        return None

    # Get title and summary using enhanced prompts for business menus
    extracted = await get_title_and_summary(content, "business_menu", business_data['name'])

    # Get embedding
    embedding = await get_embedding(content)

    # Create metadata
    metadata = {
        "source": metadata_sources,
        "source_searchable": "|".join(metadata_sources),  # For text search optimization
        "type": "business_menu",
        "business_type": business_data.get('type', []),
        "chunk_size": len(content),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }

    return ProcessedChunk(
        title=extracted['title'],
        summary=extracted['summary'],
        content=content,
        metadata=metadata,
        embedding=embedding,
        place_id=None,  # Businesses don't have a place_id
        sector_id=None,  # Businesses don't have a sector_id
        business_id=business_data['id']  # Add business_id
    )

async def process_business_for_rag(business_data: Dict, metadata_sources: List[str]) -> List[ProcessedChunk]:
    """Process business data for RAG."""
    chunks = []

    # 1. Main business info chunk
    business_chunk = await process_business_info(business_data, metadata_sources)
    chunks.append(business_chunk)

    # 2. Menu chunk (if applicable)
    menu_chunk = await process_business_menu(business_data, metadata_sources)
    if menu_chunk:
        chunks.append(menu_chunk)

    return chunks

async def generate_business_chunks_preview(business_id: str, metadata_sources: List[str]):
    """Generate chunks preview for a business without inserting them into the database."""
    # Get business data
    business_data = await get_business_data(business_id)
    if not business_data:
        logger.error(f"Could not find business with id: {business_id}")
        return None, f"Error: Could not find business with id: {business_id}"

    # Process business for RAG
    chunks = await process_business_for_rag(business_data, metadata_sources)

    return chunks, f"Generated {len(chunks)} chunks for preview. Review them before inserting to the database."

async def insert_business_chunks_to_db(chunks: List[ProcessedChunk], metadata_sources: List[str]):
    """Insert the previewed business chunks into the database."""
    if not chunks:
        return "No chunks to insert."

    # Delete existing chunks for this business_id
    business_id = chunks[0].business_id
    await delete_existing_business_chunks(business_id, metadata_sources)

    # Insert all chunks in batch
    result = await insert_chunks_batch(chunks)

    return f"Successfully inserted {len(chunks)} chunks into the database."

async def process_business(business_id: str, metadata_sources: List[str]):
    """Process a business and all its data."""
    # Get business data
    business_data = await get_business_data(business_id)
    if not business_data:
        logger.error(f"Could not find business with id: {business_id}")
        return f"Error: Could not find business with id: {business_id}"

    # Delete existing chunks for this business_id
    await delete_existing_business_chunks(business_id, metadata_sources)

    # Process business for RAG
    chunks = await process_business_for_rag(business_data, metadata_sources)

    # Insert all chunks in batch
    await insert_chunks_batch(chunks)

    return f"Successfully processed business {business_data['name']} into {len(chunks)} optimized chunks"

async def process_all_businesses(metadata_sources: List[str]):
    """Process all businesses in the database for RAG."""
    try:
        # Get all businesses
        response = supabase.table("business").select("id").execute()
        business_ids = [b['id'] for b in response.data]

        logger.info(f"Processing {len(business_ids)} businesses in batch")

        # Process each business
        all_chunks = []
        for business_id in business_ids:
            business_data = await get_business_data(business_id)
            if business_data:
                chunks = await process_business_for_rag(business_data, metadata_sources)
                all_chunks.extend(chunks)
                logger.info(f"Generated {len(chunks)} chunks for business: {business_data['name']}")

        # Delete existing chunks for all businesses first
        # This is a risky operation, so we'll confirm we have generated chunks first
        if all_chunks:
            # Use a safer approach - only delete records that have a business_id
            # This preserves any other RAG data that doesn't have a business_id (like sector data)
            # First get the distinct business_ids from the data we're about to insert
            business_ids_to_process = list(set(chunk.business_id for chunk in all_chunks if chunk.business_id))

            # Delete only the business chunks for these specific business_ids
            for business_id in business_ids_to_process:
                result = supabase.table("rag_data").delete().eq("business_id", business_id).execute()
                logger.info(f"Deleted existing chunks for business_id: {business_id}")

            # Insert all chunks in batch
            result = await insert_chunks_batch(all_chunks)
            logger.info(f"Inserted {len(all_chunks)} chunks for all businesses")

            return f"Successfully processed {len(business_ids)} businesses into {len(all_chunks)} optimized chunks"
        else:
            return "No chunks were generated. Check if there are valid businesses in the database."

    except Exception as e:
        logger.error(f"Error processing all businesses: {e}")
        return f"Error processing all businesses: {str(e)}"

async def generate_all_businesses_preview(metadata_sources: List[str]):
    """Generate preview for all businesses without inserting them."""
    try:
        # Get all businesses
        response = supabase.table("business").select("id").execute()
        business_ids = [b['id'] for b in response.data]

        logger.info(f"Generating preview for {len(business_ids)} businesses")

        # Process each business
        all_chunks = []
        for business_id in business_ids:
            business_data = await get_business_data(business_id)
            if business_data:
                chunks = await process_business_for_rag(business_data, metadata_sources)
                all_chunks.extend(chunks)
                logger.info(f"Generated {len(chunks)} chunks for business: {business_data['name']}")

        return all_chunks, f"Generated {len(all_chunks)} chunks for {len(business_ids)} businesses. Review them before inserting to the database."

    except Exception as e:
        logger.error(f"Error generating preview for all businesses: {e}")
        return None, f"Error generating preview for all businesses: {str(e)}"

def get_initial_businesses():
    """Get initial list of businesses with 'All Businesses' option."""
    try:
        response = supabase.table("business").select("id, name").order("name").execute()
        # Add an option for processing all businesses
        choices = [("All Businesses", "all")] + [(b['name'], b['id']) for b in response.data]
        logger.info(f"Initial business dropdown populated with {len(choices)} choices")
        return choices
    except Exception as e:
        logger.error(f"Error getting initial businesses: {e}")
        return [("All Businesses", "all")]  # At least return the "all" option

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
        "guadalcazar",
        "accommodation"
    ]

    # Create lists to store generated chunks for preview
    generated_sector_chunks = []
    generated_business_chunks = []
    current_sector_id = None
    current_business_id = None

    async def on_generate_sector_preview(sector_id, metadata_sources):
        # Process selected sector with selected metadata sources to generate preview
        if not metadata_sources:
            return "Error: Please select at least one metadata source", None, [], False

        nonlocal generated_sector_chunks
        nonlocal current_sector_id

        chunks, message = await generate_chunks_preview(sector_id, metadata_sources)

        if chunks:
            generated_sector_chunks = chunks
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

    async def on_generate_business_preview(business_id, metadata_sources):
        # Process selected business(es) with selected metadata sources to generate preview
        if not metadata_sources:
            return "Error: Please select at least one metadata source", None, [], False

        nonlocal generated_business_chunks
        nonlocal current_business_id

        # Handle "All Businesses" option
        if business_id == "all":
            chunks, message = await generate_all_businesses_preview(metadata_sources)
        else:
            chunks, message = await generate_business_chunks_preview(business_id, metadata_sources)

        if chunks:
            generated_business_chunks = chunks
            # Store the business_id or "all" for later use during insertion
            current_business_id = business_id

            # Prepare data for the preview table
            preview_data = []
            # For all businesses preview, limit the display to avoid overloading the UI
            display_limit = 100 if business_id == "all" else len(chunks)

            for i, chunk in enumerate(chunks[:display_limit]):
                preview_data.append([
                    i+1,
                    chunk.title,
                    chunk.summary,
                    chunk.metadata.get('type', 'Unknown'),
                    len(chunk.content)
                ])

            # Add a message about limiting preview if necessary
            if business_id == "all" and len(chunks) > display_limit:
                message += f" (Showing first {display_limit} of {len(chunks)} chunks in preview)"

            return message, gr.update(visible=True), preview_data, True
        else:
            return message, gr.update(visible=False), [], False

    async def on_insert_sector_to_db(metadata_sources):
        # Insert the previewed chunks into the database
        nonlocal generated_sector_chunks
        nonlocal current_sector_id

        if not generated_sector_chunks or not current_sector_id:
            return "No chunks to insert. Please generate a preview first."

        # Get place_id for UI display purposes
        place_id = generated_sector_chunks[0].place_id if generated_sector_chunks else None

        result = await insert_chunks_to_db(generated_sector_chunks, place_id, metadata_sources)

        # Clear the chunks after insertion
        generated_sector_chunks = []

        return result

    async def on_insert_business_to_db(metadata_sources):
        # Insert the previewed business chunks into the database
        nonlocal generated_business_chunks
        nonlocal current_business_id

        if not generated_business_chunks:
            return "No chunks to insert. Please generate a preview first."

        # Special handling for "all" businesses case
        if current_business_id == "all":
            # Confirm we have chunks before this potentially destructive operation
            if generated_business_chunks:
                # Use a safer approach - only delete records that have a business_id
                # This preserves any other RAG data that doesn't have a business_id (like sector data)
                # First get the distinct business_ids from the data we're about to insert
                business_ids_to_process = list(set(chunk.business_id for chunk in generated_business_chunks if chunk.business_id))

                # Delete only the business chunks for these specific business_ids
                for business_id in business_ids_to_process:
                    result = supabase.table("rag_data").delete().eq("business_id", business_id).execute()
                    logger.info(f"Deleted existing chunks for business_id: {business_id}")

                # Insert all chunks in batch
                result = await insert_chunks_batch(generated_business_chunks)

                count_inserted = len(generated_business_chunks)
                logger.info(f"Inserted {count_inserted} chunks for all businesses")

                # Clear the chunks after insertion
                generated_business_chunks = []

                return f"Successfully inserted {count_inserted} chunks for all businesses"
        else:
            # Regular single business processing
            business_id = generated_business_chunks[0].business_id
            result = await insert_business_chunks_to_db(generated_business_chunks, metadata_sources)

            # Clear the chunks after insertion
            generated_business_chunks = []

            return result

    async def update_sector_dropdown():
        sectors = await get_all_sectors()
        choices = [(s['name'], s['id']) for s in sectors]
        logger.info(f"Updating dropdown with {len(choices)} sector choices")
        return choices

    async def update_business_dropdown():
        businesses = await get_all_businesses()
        # Add an option for processing all businesses
        choices = [("All Businesses", "all")] + [(b['name'], b['id']) for b in businesses]
        logger.info(f"Updating dropdown with {len(choices)} business choices")
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

    async def update_business_type(business_id):
        """Update the business type tag when a business is selected."""
        if not business_id:
            return "N/A"

        # Handle the "All Businesses" option
        if business_id == "all":
            return "Procesando todos los negocios"

        business_data = await get_business_data(business_id)
        if not business_data or not business_data.get('type'):
            return "Tipo: No especificado"

        # business type is stored as a JSON array
        try:
            business_types = business_data['type']
            if isinstance(business_types, str):
                business_types = json.loads(business_types)

            if not business_types:
                return "Tipo: No especificado"

            types_str = ", ".join(business_types)
            return f"Tipo: {types_str}"
        except Exception as e:
            logger.error(f"Error parsing business type: {e}")
            return "Tipo: Error"

    # Get sectors and businesses synchronously for initial dropdown population
    def get_initial_sectors():
        response = supabase.table("sector").select("id, name").order("name").execute()
        choices = [(s['name'], s['id']) for s in response.data]
        logger.info(f"Initial sector dropdown populated with {len(choices)} choices")
        return choices

    # Get initial businesses with the "All Businesses" option
    initial_sectors = get_initial_sectors()
    initial_businesses = get_initial_businesses()

    with gr.Blocks(title="RAG Data Update Tool") as app:
        gr.Markdown("# RAG Data Update Tool")

        with gr.Tabs():
            with gr.TabItem("Sector Data"):
                gr.Markdown("## Select a climbing sector to process for RAG data")

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

                refresh_sector_btn = gr.Button("Refresh Sector List")
                refresh_sector_btn.click(fn=update_sector_dropdown, outputs=[sector_dropdown])

                # Replace text input with multiselect dropdown
                sector_metadata_sources = gr.Dropdown(
                    label="Metadata Sources",
                    choices=metadata_options,
                    multiselect=True,
                    value=[],
                    info="Select metadata sources for the chunks"
                )

                # Preview button instead of direct processing
                sector_preview_btn = gr.Button("Generate Preview")
                sector_preview_result = gr.Textbox(label="Preview Result", interactive=False)

                # Preview table container that is initially hidden
                with gr.Column(visible=False) as sector_preview_container:
                    gr.Markdown("### Generated Chunks Preview")
                    sector_preview_table = gr.Dataframe(
                        headers=["#", "Title", "Summary", "Type", "Content Length"],
                        datatype=["number", "str", "str", "str", "number"],
                        label="Chunks Preview"
                    )

                    # Insert button only visible after preview is generated
                    sector_insert_btn = gr.Button("Insert to Database", variant="primary")
                    sector_insert_result = gr.Textbox(label="Insert Result", interactive=False)

                # Connect preview button to generate preview function
                sector_preview_btn.click(
                    fn=on_generate_sector_preview,
                    inputs=[sector_dropdown, sector_metadata_sources],
                    outputs=[sector_preview_result, sector_preview_container, sector_preview_table, sector_insert_btn]
                )

                # Connect insert button to insert function
                sector_insert_btn.click(
                    fn=on_insert_sector_to_db,
                    inputs=[sector_metadata_sources],
                    outputs=[sector_insert_result]
                )

            with gr.TabItem("Business Data"):
                gr.Markdown("## Select a business to process for RAG data")

                # Load businesses on startup with pre-fetched data
                business_dropdown = gr.Dropdown(
                    label="Select Business",
                    choices=initial_businesses,
                    info="Choose a business to process",
                    scale=1
                )

                # Add business type tag
                business_type_tag = gr.Textbox(
                    label="Business Type",
                    value="Tipo: No seleccionado",
                    interactive=False
                )

                # Update business type when business selection changes
                business_dropdown.change(
                    fn=update_business_type,
                    inputs=[business_dropdown],
                    outputs=[business_type_tag]
                )

                refresh_business_btn = gr.Button("Refresh Business List")
                refresh_business_btn.click(fn=update_business_dropdown, outputs=[business_dropdown])

                # Metadata sources for business
                business_metadata_sources = gr.Dropdown(
                    label="Metadata Sources",
                    choices=metadata_options,
                    multiselect=True,
                    value=["accommodation"],  # Default to "accommodation" for business data
                    info="Select metadata sources for the chunks"
                )

                # Preview button for business data
                business_preview_btn = gr.Button("Generate Preview")
                business_preview_result = gr.Textbox(label="Preview Result", interactive=False)

                # Preview table container for business data
                with gr.Column(visible=False) as business_preview_container:
                    gr.Markdown("### Generated Business Chunks Preview")
                    business_preview_table = gr.Dataframe(
                        headers=["#", "Title", "Summary", "Type", "Content Length"],
                        datatype=["number", "str", "str", "str", "number"],
                        label="Chunks Preview"
                    )

                    # Insert button only visible after preview is generated
                    business_insert_btn = gr.Button("Insert to Database", variant="primary")
                    business_insert_result = gr.Textbox(label="Insert Result", interactive=False)

                # Connect preview button to generate preview function for business
                business_preview_btn.click(
                    fn=on_generate_business_preview,
                    inputs=[business_dropdown, business_metadata_sources],
                    outputs=[business_preview_result, business_preview_container, business_preview_table, business_insert_btn]
                )

                # Connect insert button to insert function for business
                business_insert_btn.click(
                    fn=on_insert_business_to_db,
                    inputs=[business_metadata_sources],
                    outputs=[business_insert_result]
                )

    return app

async def main():
    # Add a description to logging
    logger.info("Starting RAG data update tool with business and boulder support")
    app = create_ui()
    app.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    # Run the async app inside asyncio event loop
    asyncio.run(main())
