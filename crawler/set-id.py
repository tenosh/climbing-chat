import os
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

# Configuration
FILE_NAME = "Gruta de las Candelas.md"
BASE_PATH = "./climbs/san_luis_potosi/guadalcazar"
AREA_ID = "812691b8-25a4-4817-86c8-369979149d14"  # Area ID for Gruta de las Candelas

class RouteIdUpdater:
    def __init__(self, base_path: str = BASE_PATH):
        self.base_path = base_path
        self.file_path = os.path.join(base_path, FILE_NAME)

    def read_md_file(self) -> Optional[str]:
        """Read content from the markdown file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {self.file_path}: {str(e)}")
            return None

    def write_md_file(self, content: str) -> bool:
        """Write updated content to the markdown file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            return True
        except Exception as e:
            print(f"Error writing to file {self.file_path}: {str(e)}")
            return False

    async def get_route_ids(self) -> Dict[str, str]:
        """Fetch route IDs from the database by name."""
        try:
            result = supabase.table("route").select("id, name").eq("areaId", AREA_ID).execute()

            if not result.data:
                print(f"No routes found for area ID: {AREA_ID}")
                return {}

            route_ids = {route["name"]: route["id"] for route in result.data}
            print(f"Found {len(route_ids)} routes in the database")
            # Print all route names and IDs found
            for name, id in route_ids.items():
                print(f"  - {name}: {id}")
            return route_ids

        except Exception as e:
            print(f"Error fetching route IDs: {e}")
            return {}

    def update_markdown_with_ids(self, content: str, route_ids: Dict[str, str]) -> str:
        """Update the markdown content with route IDs."""
        updated_content = content

        # Find all route sections in the markdown
        # Finding all routes that start with ### and may include (Ruta)
        route_sections = re.split(r'(?=^### )', content, flags=re.MULTILINE)

        total_routes = 0
        updated_routes = 0

        for section in route_sections:
            if not section.strip() or not section.startswith('###'):
                continue

            total_routes += 1

            # Extract route name
            name_match = re.match(r'^### (.+?)(?:\s*\(Ruta\))?$', section, re.MULTILINE)
            if not name_match:
                continue

            route_name = name_match.group(1).strip()
            print(f"Processing route: {route_name}")

            # Check if route has an ID in our dictionary
            if route_name in route_ids:
                route_id = route_ids[route_name]

                # Check if ID is already in metadata
                if "**ID:**" in section:
                    # Update existing ID
                    updated_section = re.sub(
                        r'- \*\*ID:\*\* .+?\n',
                        f'- **ID:** {route_id}\n',
                        section
                    )
                else:
                    # Add ID after the route header
                    lines = section.split('\n')
                    header_index = 0
                    for i, line in enumerate(lines):
                        if line.startswith('###'):
                            header_index = i
                            break

                    # Insert ID after header and before other content
                    if header_index + 1 < len(lines) and lines[header_index + 1].strip() == '':
                        # There's already a blank line after header
                        lines.insert(header_index + 2, f'- **ID:** {route_id}')
                    else:
                        # No blank line, add one and then the ID
                        lines.insert(header_index + 1, '')
                        lines.insert(header_index + 2, f'- **ID:** {route_id}')

                    updated_section = '\n'.join(lines)

                # Replace the old section with updated section in the content
                updated_content = updated_content.replace(section, updated_section)
                updated_routes += 1
                print(f"  Updated route: {route_name} with ID: {route_id}")
            else:
                print(f"  Warning: No ID found for route: {route_name}")

        print(f"Total routes found in markdown: {total_routes}")
        print(f"Total routes updated: {updated_routes}")
        return updated_content

    async def update_routes_with_ids(self) -> None:
        """Main method to update routes with IDs."""
        print(f"Processing file: {self.file_path}")

        # Read the markdown file
        content = self.read_md_file()
        if not content:
            return

        # Get route IDs from the database
        route_ids = await self.get_route_ids()
        if not route_ids:
            print("No route IDs found, aborting update")
            return

        # Update markdown with IDs
        updated_content = self.update_markdown_with_ids(content, route_ids)

        # Write the updated content back to the file
        if self.write_md_file(updated_content):
            print(f"Successfully updated {FILE_NAME} with route IDs")
        else:
            print(f"Failed to update {FILE_NAME}")

async def main():
    updater = RouteIdUpdater()
    await updater.update_routes_with_ids()

if __name__ == "__main__":
    asyncio.run(main())
