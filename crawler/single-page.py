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
        "https://www.thecrag.com/en/climbing/mexico/area/289682910"
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

    # Parse the JSON string if it's a string
    if isinstance(data, str):
        data = json.loads(data)

    # Clean the data by removing empty objects
    cleaned_data = clean_data(data)

    # Create a safe filename from the URL
    filename = urlparse(url).path.strip('/').replace('/', '_') + '.json'
    # Ensure the filename is valid
    filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"Saved content to: {filename}")

async def save_to_supabase(url: str, data: Any):
    """Save crawled content to Supabase database after cleaning empty objects."""
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
        # Handle the case where cleaned_data is a list
        if isinstance(cleaned_data, list):
            for area_data in cleaned_data:
                area_name = area_data.get('area_name', '')

                # Check if area already exists
                existing_area = supabase.table('Area').select('*').eq('name', area_name).execute()

                area_record = {
                    'name': area_name,
                    'description': area_data.get('area_description', ''),
                    'approach': area_data.get('area_approach', ''),
                    'ethic': area_data.get('area_ethic', '')
                }

                if not existing_area.data:
                    # Create new area if it doesn't exist
                    area_record['id'] = str(uuid.uuid4())
                    area_response = supabase.table('Area').insert(area_record).execute()
                    area_id = area_response.data[0]['id']
                    print(f"Created new area: {area_name}")
                else:
                    # Update existing area if content has changed
                    area_id = existing_area.data[0]['id']
                    if any(existing_area.data[0].get(k) != v for k, v in area_record.items()):
                        supabase.table('Area').update(area_record).eq('id', area_id).execute()
                        print(f"Updated existing area: {area_name}")
                    else:
                        print(f"Area {area_name} unchanged, skipping update")

                # Process and insert/update routes
                routes = area_data.get('routes', [])
                for route in routes:
                    route_name = route.get('name', '')

                    # Check if route already exists in this area
                    existing_route = supabase.table('Route').select('*').eq('name', route_name).eq('areaId', area_id).execute()

                    route_data = {
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

                    if not existing_route.data:
                        # Create new route if it doesn't exist
                        route_data['id'] = str(uuid.uuid4())
                        supabase.table('Route').insert(route_data).execute()
                        print(f"Created new route: {route_name}")
                    else:
                        # Update existing route if content has changed
                        route_id = existing_route.data[0]['id']
                        if any(existing_route.data[0].get(k) != v for k, v in route_data.items() if k != 'createdBy'):
                            supabase.table('Route').update(route_data).eq('id', route_id).execute()
                            print(f"Updated existing route: {route_name}")
                        else:
                            print(f"Route {route_name} unchanged, skipping update")

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