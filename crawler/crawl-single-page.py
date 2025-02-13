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
        # "https://www.thecrag.com/en/climbing/mexico/area/424260063",
        # "https://www.thecrag.com/en/climbing/mexico/area/289682910",
        # "https://www.thecrag.com/en/climbing/mexico/area/2263518714",
        # "https://www.thecrag.com/en/climbing/mexico/area/2172540612",
        "https://www.thecrag.com/en/climbing/mexico/area/2246283312",
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
        #     options={"ignore_images": True}
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
                    save_content(url, result.extracted_content, format="markdown")
                    # await save_to_supabase(url, result.extracted_content)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def save_content(url: str, data: Any, format: str = "json"):
    """
    Save crawled content to either JSON or Markdown file.

    Args:
        url: The source URL
        data: The crawled data
        format: Output format - either "json" or "markdown" (default: "json")
    """
    save_dir = "./climbs/san_luis_potosi/guadalcazar"
    os.makedirs(save_dir, exist_ok=True)

    if format.lower() == "json":
        def clean_data(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if v not in ({}, [], None, ""):
                        if k == 'quality' and isinstance(v, str):
                            # Extract only numbers from quality string
                            cleaned[k] = ''.join(filter(str.isdigit, v))
                        elif k == 'length' and isinstance(v, str):
                            # Extract only the first number from length string
                            numbers = ''.join(filter(str.isdigit, v.split(',')[0]))
                            cleaned[k] = numbers if numbers else v
                        else:
                            cleaned[k] = clean_data(v)
                return cleaned
            elif isinstance(obj, list):
                return [clean_data(item) for item in obj if item not in ({}, [], None, "")]
            return obj

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

        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    elif format.lower() == "markdown":
        # Parse JSON if it's a string
        if isinstance(data, str):
            data = json.loads(data)

        # Clean empty values
        def clean_data(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if v not in ({}, [], None, ""):
                        if k == 'quality' and isinstance(v, str):
                            # Extract only numbers from quality string
                            cleaned[k] = ''.join(filter(str.isdigit, v))
                        elif k == 'length' and isinstance(v, str):
                            # Extract only the first number from length string
                            numbers = ''.join(filter(str.isdigit, v.split(',')[0]))
                            cleaned[k] = numbers if numbers else v
                        else:
                            cleaned[k] = clean_data(v)
                return cleaned
            elif isinstance(obj, list):
                return [clean_data(item) for item in obj if item not in ({}, [], None, "")]
            return obj

        data = clean_data(data)

        # Create markdown content
        def generate_markdown(area_data):
            markdown = []

            # Area information
            if area_data.get('area_name'):
                markdown.append(f"# {area_data['area_name']} (Area) \n")

            if area_data.get('area_description'):
                markdown.append("## Description (Description of Area)\n")
                markdown.append(f"{area_data['area_description']}\n")

            if area_data.get('area_approach'):
                markdown.append("## Approach (Approach to Area)\n")
                markdown.append(f"{area_data['area_approach']}\n")

            if area_data.get('area_ethic'):
                markdown.append("## Ethics (Ethics of Area)\n")
                markdown.append(f"{area_data['area_ethic']}\n")

            # Routes section
            if area_data.get('routes'):
                markdown.append("## Routes (Routes of Area) \n")
                for route in area_data['routes']:
                    if route.get('name'):
                        markdown.append(f"### {route['name']}")

                        # Route details in a list format
                        details = []
                        if route.get('grade'):
                            details.append(f"- **Grade:** {route['grade']}")
                        if route.get('type'):
                            details.append(f"- **Type:** {route['type']}")
                        if route.get('quality'):
                            quality = route['quality'].replace('Quality: ', '')
                            details.append(f"- **Quality:** {quality}")
                        if route.get('length'):
                            details.append(f"- **Length:** {route['length']}")
                        if route.get('bolts'):
                            details.append(f"- **Bolts:** {route['bolts']}")

                        if details:
                            markdown.append('\n' + '\n'.join(details))

                        if route.get('description'):
                            markdown.append(f"\n{route['description']}")

                        markdown.append('\n')  # Add space between routes

            return '\n'.join(markdown)

        # Generate markdown content
        if isinstance(data, list):
            content = '\n'.join(generate_markdown(area) for area in data)
        else:
            content = generate_markdown(data)

        # Create filename from URL - use a fixed filename
        filename = "Panales.md"
        filepath = os.path.join(save_dir, filename)

        # Add metadata header
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        markdown_content = f"""---
url: {url}
date_crawled: {current_date}
---

{content}

---
"""  # Added separator between entries

        # Append content if file exists, otherwise create new file
        mode = 'a' if os.path.exists(filepath) else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(markdown_content)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'markdown'.")

    print(f"Saved content to: {filepath}")

async def save_to_supabase(url: str, data: Any):
    """Save crawled content to Supabase database, only inserting new routes."""
    def clean_data(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if v not in ({}, [], None, "", " "):  # Added space to empty checks
                    if k == 'quality' and isinstance(v, str):
                        # Extract only numbers from quality string and convert to int
                        numbers = ''.join(filter(str.isdigit, v))
                        cleaned[k] = int(numbers) if numbers else None
                    elif k == 'length' and isinstance(v, str):
                        # Extract only the first number from length string and convert to int
                        numbers = ''.join(filter(str.isdigit, v.split(',')[0]))
                        cleaned[k] = int(numbers) if numbers else None
                    elif k == 'bolts' and isinstance(v, str):
                        # Extract numbers from bolts string and convert to int
                        numbers = ''.join(filter(str.isdigit, v))
                        cleaned[k] = int(numbers) if numbers else None
                    else:
                        cleaned[k] = clean_data(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj if item not in ({}, [], None, "", " ")]
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
                        # Convert numeric fields explicitly, ensuring None for invalid values
                        try:
                            quality = int(''.join(filter(str.isdigit, str(route.get('quality', ''))))) if route.get('quality') else None
                        except ValueError:
                            quality = None

                        try:
                            length = int(''.join(filter(str.isdigit, str(route.get('length', '')).split(',')[0]))) if route.get('length') else None
                        except ValueError:
                            length = None

                        try:
                            bolts = int(''.join(filter(str.isdigit, str(route.get('bolts', ''))))) if route.get('bolts') else None
                        except ValueError:
                            bolts = None

                        route_data = {
                            'id': str(uuid.uuid4()),
                            'name': route_name,
                            'description': route.get('description', ''),
                            'grade': route.get('grade', ''),
                            'quality': quality,
                            'type': route.get('type', ''),
                            'bolts': bolts,
                            'length': length,
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