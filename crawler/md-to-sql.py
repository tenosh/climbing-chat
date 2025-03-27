import os
import sys
import re
import asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configuration - change these variables to process different files
FILE_NAME = "San Cayetano.md"  # Set the file name here
SECTOR_ID = "14210743-d8a6-446b-93c8-5c867943ea40"  # Area ID for San Cayetano

class RouteExtractor:
    def __init__(self, base_path: str = "./climbs/san_luis_potosi/guadalcazar"):
        self.base_path = base_path

    def read_md_file(self, filename: str) -> Optional[str]:
        """Read content from a markdown file."""
        file_path = os.path.join(self.base_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None

    def extract_area_name(self, content: str) -> str:
        """Extract the area name from the markdown content."""
        # Look for the first heading which should be the area name
        match = re.search(r'^# (.+?)(?:\s*\(|$)', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Unknown Area"

    def extract_routes_section(self, content: str) -> Optional[str]:
        """Extract the routes section from the markdown content."""
        print("Looking for routes section in content...")

        # Check if the pattern exists in the content
        if "## Rutas" in content:
            print("Found '## Rutas' in content")
        else:
            print("WARNING: '## Rutas' not found in content")

        # Simpler approach: find the "## Rutas" line and take everything after it
        lines = content.split('\n')
        routes_start_idx = -1

        for i, line in enumerate(lines):
            if line.startswith("## Rutas"):
                routes_start_idx = i
                break

        if routes_start_idx == -1:
            print("Routes section not found in the file")
            return None

        # Find the next section heading (if any)
        next_section_idx = len(lines)
        for i in range(routes_start_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                next_section_idx = i
                break

        # Extract the routes section
        routes_section = '\n'.join(lines[routes_start_idx + 1:next_section_idx]).strip()

        if not routes_section:
            print("Routes section is empty")
            return None

        print(f"Found routes section with {len(routes_section)} characters")
        return routes_section

    def parse_routes(self, routes_section: str) -> List[Dict[str, Any]]:
        """Parse individual routes from the routes section."""
        routes = []

        # Split by route headers (### Route Name)
        route_blocks = re.split(r'(?=^### )', routes_section, flags=re.MULTILINE)

        for block in route_blocks:
            if not block.strip() or not block.startswith('###'):
                continue

            route = {
                "name": "",
                "description": "",
                "grade": None,
                "bolts": None,
                "length": None,
                "quality": None,
                "type": None,
                "sector_id": SECTOR_ID
            }

            # Extract route name
            name_match = re.match(r'^### (.+?)(?:\s*\(Ruta\))?$', block, re.MULTILINE)
            if name_match:
                route["name"] = name_match.group(1).strip()

            # Extract grade
            grade_match = re.search(r'\*\*Grado:\*\* ([\w\d.+/-]+)', block)
            if grade_match:
                route["grade"] = grade_match.group(1).strip()

            # Extract type
            type_match = re.search(r'\*\*Tipo:\*\* (.+?)(?:\s+\*\*|\s*$|\n)', block)
            if type_match:
                route["type"] = type_match.group(1).strip().lower()

            # Extract quality
            quality_match = re.search(r'\*\*Calidad:\*\* (\d+)', block)
            if quality_match:
                route["quality"] = int(quality_match.group(1))

            # Extract length
            length_match = re.search(r'\*\*Longitud:\*\* (\d+)\s*metros', block)
            if length_match:
                route["length"] = int(length_match.group(1))

            # Extract bolts
            bolts_match = re.search(r'\*\*Bolts:\*\* (\d+)', block)
            if bolts_match:
                route["bolts"] = int(bolts_match.group(1))

            # Extract description - find the actual description text after all metadata
            lines = block.split('\n')
            description_lines = []
            metadata_ended = False

            # Skip the first line (route name) and process the rest
            for line in lines[1:]:
                # Check if this is a metadata line (starts with - **)
                if line.strip().startswith('- **'):
                    continue  # Skip metadata lines

                # If we have a non-empty line that isn't metadata, we've found the description
                if line.strip() and not line.strip().startswith('- **'):
                    description_lines.append(line)

            route["description"] = '\n'.join(description_lines).strip()

            routes.append(route)

        return routes

    async def insert_routes(self, routes: List[Dict[str, Any]]) -> None:
        """Insert routes into the database."""
        for route in routes:
            try:
                # Check if route already exists
                existing = supabase.table("route").select("id").eq("sector_id", SECTOR_ID).eq("name", route["name"]).execute()

                if existing.data:
                    print(f"Route '{route['name']}' already exists, skipping...")
                else:
                    print(f"Inserting route: {route['name']}")
                    result = supabase.table("route").insert(route).execute()

            except Exception as e:
                print(f"Error inserting route '{route['name']}': {e}")

    async def process_file(self) -> None:
        """Process the specified markdown file."""
        content = self.read_md_file(FILE_NAME)
        if not content:
            return

        area_name = self.extract_area_name(content)
        print(f"Processing area: {area_name} from file: {FILE_NAME}")

        routes_section = self.extract_routes_section(content)
        if not routes_section:
            print("Routes section not found in the file")
            return

        routes = self.parse_routes(routes_section)
        print(f"Found {len(routes)} routes in {FILE_NAME}")

        # Print detailed route information before insertion
        # print("\n=== EXTRACTED ROUTES ===")
        # for i, route in enumerate(routes, 1):
        #     print(f"\nRoute {i}: {route['name']}")
        #     print(f"  Grade: {route['grade']}")
        #     print(f"  Type: {route['type']}")
        #     print(f"  Quality: {route['quality']}")
        #     print(f"  Length: {route['length']} meters" if route['length'] else "  Length: Not specified")
        #     print(f"  Bolts: {route['bolts']}" if route['bolts'] else "  Bolts: Not specified")
        #     print(f"  Description: {route['description']}")
        # print("=======================\n")

        await self.insert_routes(routes)
        print(f"Finished processing {FILE_NAME}")

async def main():
    extractor = RouteExtractor()
    await extractor.process_file()

if __name__ == "__main__":
    asyncio.run(main())
