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
import uuid

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def get_thecrag_urls() -> List[str]:
    """Get URLs to crawl."""
    return [
        "https://www.thecrag.com/en/climbing/mexico/area/424260063",
        # "https://www.thecrag.com/en/climbing/mexico/area/289682910",
        # "https://www.thecrag.com/en/climbing/mexico/area/2263518714",
        # "https://www.thecrag.com/en/climbing/mexico/area/2172540612",
        # "https://www.thecrag.com/en/climbing/mexico/area/2246283312",
    ]

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    schema = {
        "name": "climbing_area",
        "baseSelector": ".regions__content",
        "fields": [
            {
                "name": "area_name",
                "selector": ".heading__t",
                "type": "text"
            },
            {
                "name": "area_description",
                "selector": ".description .content .markdown",
                "type": "text"
            },
            {
                "name": "area_approach",
                "selector": ".approach .content .markdown",
                "type": "text"
            },
            {
                "name": "area_ethic",
                "selector": ".ethic .content .markdown",
                "type": "text"
            },
            {
                "name": "routes",
                "selector": ".route",
                "type": "nested_list",    # repeated sub-objects
                "fields": [
                    {
                        "name": "name",
                        "selector": ".primary-node-name",
                        "type": "text"
                    },
                    {
                        "name": "description",
                        "selector": ".desc p",
                        "type": "text"
                    },
                    {
                        "name": "grade",
                        "selector": ".r-grade span",
                        "type": "text"
                    },
                    {
                        "name": "quality",
                        "selector": ".name a span",
                        "type": "attribute",
                        "attribute": "title"
                    },
                    {
                        "name": "type",
                        "selector": ".flags .tags",
                        "type": "text",
                    },
                    {
                        "name": "bolts",
                        "selector": ".flags .bolts",
                        "type": "text",
                    },
                    {
                        "name": "length",
                        "selector": ".flags .attr",
                        "type": "text",
                    },
                ]
            }
        ]
    }
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_external_images=True,
        excluded_tags=['form', 'header', 'footer', 'nav', 'script', 'style'],
        css_selector=".regions__content",
        extraction_strategy=extraction_strategy,
        # markdown_generator=DefaultMarkdownGenerator(
        #     content_filter=PruningContentFilter(threshold=0.6),
        #     options={"ignore_links": True, "ignore_images": True}
        # )
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    save_to_json(url, result.extracted_content)
                    await save_to_supabase(url, result.extracted_content)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def save_to_markdown(url: str, content: str):
    """Save crawled content to a markdown file."""
    # Create a safe filename from the URL
    filename = urlparse(url).path.strip('/').replace('/', '_') + '.md'
    # Ensure the filename is valid
    filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved content to: {filename}")

def save_to_json(url: str, data: Any):
    """Save crawled content to a JSON file after cleaning empty objects."""
    def clean_data(obj):
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items() if v not in ({}, [], None, "")}
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj if item not in ({}, [], None, "")]
        return obj

    # Create the directory if it doesn't exist
    save_dir = "./climbs/san_luis_potosi/guadalcazar"
    os.makedirs(save_dir, exist_ok=True)

    # Parse the JSON string if it's a string
    if isinstance(data, str):
        data = json.loads(data)

    # Clean the data by removing empty objects
    cleaned_data = clean_data(data)

    # Extract area name from the data and create a safe version
    area_name = ""
    if isinstance(cleaned_data, list) and cleaned_data:
        area_name = cleaned_data[0].get('area_name', '')
    elif isinstance(cleaned_data, dict):
        area_name = cleaned_data.get('area_name', '')

    safe_area_name = ''.join(c for c in area_name.lower() if c.isalnum() or c in ('_', '-', ' '))
    safe_area_name = safe_area_name.replace(' ', '_')

    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Create filename with area name and date
    filename = f"{safe_area_name}_{current_date}.json"
    if not safe_area_name:  # Fallback to URL-based name if no area name found
        filename = urlparse(url).path.strip('/').replace('/', '_') + '.json'
        filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))

    # Join the directory path with the filename
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"Saved content to: {filepath}")

async def save_to_supabase(url: str, data: Any):
    """Save crawled content to Supabase database, only inserting new routes."""
    def clean_data(obj):
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items() if v not in ({}, [], None, "")}
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj if item not in ({}, [], None, "")]
        return obj

    # Parse the JSON string if it's a string
    if isinstance(data, str):
        data = json.loads(data)

    # Clean the data
    cleaned_data = clean_data(data)

    try:
        if isinstance(cleaned_data, list):
            for area_data in cleaned_data:
                area_name = area_data.get('area_name', '')

                # Prepare area record
                area_record = {
                    'name': area_name,
                    'description': area_data.get('area_description', ''),
                    'approach': area_data.get('area_approach', ''),
                    'ethic': area_data.get('area_ethic', '')
                }

                # Get all existing areas in one query
                existing_areas = supabase.table('Area').select('*').execute()
                area_map = {area['name']: area for area in existing_areas.data}

                # Check if area exists
                if area_name not in area_map:
                    # Create new area
                    area_record['id'] = str(uuid.uuid4())
                    area_response = supabase.table('Area').insert(area_record).execute()
                    area_id = area_response.data[0]['id']
                    print(f"Created new area: {area_name}")
                else:
                    # Use existing area id without updating
                    area_id = area_map[area_name]['id']
                    print(f"Using existing area: {area_name}")

                # Get all existing routes for this area in one query
                existing_routes = supabase.table('Route').select('*').eq('areaId', area_id).execute()
                existing_route_names = {route['name'] for route in existing_routes.data}

                # Prepare new routes for insertion
                routes_to_insert = []
                for route in area_data.get('routes', []):
                    route_name = route.get('name', '')
                    if route_name and route_name not in existing_route_names:
                        route_data = {
                            'id': str(uuid.uuid4()),
                            'name': route_name,
                            'description': route.get('description', ''),
                            'grade': route.get('grade', ''),
                            'quality': route.get('quality', ''),
                            'type': route.get('type', ''),
                            'bolts': route.get('bolts', ''),
                            'length': route.get('length', ''),
                            'areaId': area_id,
                            'createdBy': 'crawler'
                        }
                        routes_to_insert.append(route_data)

                # Batch insert new routes
                if routes_to_insert:
                    supabase.table('Route').insert(routes_to_insert).execute()
                    print(f"Created {len(routes_to_insert)} new routes")
                else:
                    print("No new routes to add")

        print(f"Successfully processed area and routes from: {url}")

    except Exception as e:
        print(f"Error saving data from {url} to Supabase: {str(e)}")

async def main():
    # Get URLs from Pydantic AI docs
    urls = get_thecrag_urls()
    if not urls:
        print("No URLs found to crawl")
        return

    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())