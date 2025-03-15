import os
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from PIL import Image
import io
import requests
from pathlib import Path
import uuid
import re

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configuration
BASE_PATH = "./climbs/san_luis_potosi/guadalcazar/boulders.xlsx"
AREA_ID = "5f08920b-ff8b-45ed-b3f8-a4976bdd71b7"
DEFAULT_DESCRIPTION = "Descripción temporal del boulder"
DEFAULT_QUALITY = 100
DEFAULT_TYPE = ["boulder"]
BUCKET_NAME = "cactux"
STORAGE_FOLDER = "boulders"

class BoulderImporter:
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path

    def read_excel_file(self) -> pd.DataFrame:
        """Read data from the Excel file."""
        try:
            df = pd.read_excel(self.excel_file_path)
            print(f"Successfully read {len(df)} rows from {self.excel_file_path}")
            return df
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            raise

    def process_image(self, image_path: str) -> Optional[str]:
        """
        Process an image:
        1. Crop to 4:3 aspect ratio
        2. Optimize
        3. Convert to .webp format
        Returns the path to the processed image in Supabase storage
        """
        if not image_path or pd.isna(image_path):
            return None

        try:
            # Handle both local files and URLs
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            else:
                img = Image.open(image_path)

            # Calculate dimensions for 4:3 aspect ratio crop
            width, height = img.size
            target_ratio = 4/3

            if width/height > target_ratio:
                # Image is wider than 4:3, crop width
                new_width = int(height * target_ratio)
                left = (width - new_width) // 2
                img = img.crop((left, 0, left + new_width, height))
            else:
                # Image is taller than 4:3, crop height
                new_height = int(width / target_ratio)
                top = (height - new_height) // 2
                img = img.crop((0, top, width, top + new_height))

            # Create a temporary file
            unique_id = str(uuid.uuid4())
            temp_path = f"temp_{unique_id}.webp"

            # Save as webp with optimization
            img.save(temp_path, format="WEBP", quality=85, optimize=True)

            # Upload to Supabase storage
            storage_path = f"{STORAGE_FOLDER}/{unique_id}.webp"
            with open(temp_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "image/webp"}
                )

            # Clean up temp file
            os.remove(temp_path)

            # Return the public URL
            return supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def prepare_boulder_data(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a row of Excel data to a boulder database record."""
        # Process image if available
        image_url = self.process_image(row.get(6))  # Column 7 (index 6) is the image

        # Convert coordinates from DMS to decimal degrees if needed
        lat = row.get(3)
        lon = row.get(4)

        if pd.notna(lat) and isinstance(lat, str) and ('°' in lat or "'" in lat or '"' in lat):
            lat = self.dms_to_decimal(lat)
        elif pd.notna(lat):
            lat = float(lat)
        else:
            lat = None

        if pd.notna(lon) and isinstance(lon, str) and ('°' in lon or "'" in lon or '"' in lon):
            lon = self.dms_to_decimal(lon)
        elif pd.notna(lon):
            lon = float(lon)
        else:
            lon = None

        boulder = {
            "name": row.get(0),  # Column 1 (index 0) is the name
            "grade": row.get(1),  # Column 2 (index 1) is the grade
            "description": row.get(2) if pd.notna(row.get(2)) else DEFAULT_DESCRIPTION,  # Column 3 (index 2) is the description
            "latitude": lat,  # Using converted latitude
            "longitude": lon,  # Using converted longitude
            "height": row.get(5) if pd.notna(row.get(5)) else None,  # Column 6 (index 5) is height
            "image": image_url,
            "areaId": AREA_ID,
            "quality": DEFAULT_QUALITY,
            "type": DEFAULT_TYPE
        }

        return boulder

    def dms_to_decimal(self, dms_str: str) -> float:
        """Convert coordinates from DMS (Degrees, Minutes, Seconds) to decimal degrees."""
        # Remove any spaces that might be in the string
        dms_str = dms_str.strip()

        # Check for hemisphere (N/S/E/W)
        hemisphere = 1
        if dms_str.endswith('S') or dms_str.endswith('W'):
            hemisphere = -1

        # Remove the hemisphere letter
        if dms_str[-1] in 'NSEW':
            dms_str = dms_str[:-1]

        # Try to parse using regex
        # This pattern looks for degrees, minutes, and seconds with various separators
        pattern = r'(\d+)°\s*(\d+)[\'′]?\s*(\d+\.?\d*)[\"″]?'
        match = re.search(pattern, dms_str)

        if match:
            degrees = float(match.group(1))
            minutes = float(match.group(2))
            seconds = float(match.group(3))

            # Convert to decimal degrees: DD = d + m/60 + s/3600
            decimal = degrees + minutes/60 + seconds/3600
            return decimal * hemisphere

        # If the regex pattern didn't match, try a simpler approach
        # Split by degree symbol, minute symbol, and second symbol
        parts = re.split(r'[°\'\"″′]', dms_str)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) >= 3:  # We have degrees, minutes, and seconds
            degrees = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            decimal = degrees + minutes/60 + seconds/3600
            return decimal * hemisphere
        elif len(parts) == 2:  # We have degrees and minutes
            degrees = float(parts[0])
            minutes = float(parts[1])
            decimal = degrees + minutes/60
            return decimal * hemisphere

        # If all else fails, try to convert directly to float
        try:
            return float(dms_str) * hemisphere
        except ValueError:
            print(f"Could not parse coordinate: {dms_str}")
            return None

    async def insert_boulder(self, boulder: Dict[str, Any]) -> None:
        """Insert a boulder into the database."""
        try:
            # Check if boulder already exists with the same name in this area
            existing = supabase.table("boulder").select("id").eq("areaId", AREA_ID).eq("name", boulder["name"]).execute()

            if existing.data:
                print(f"Boulder '{boulder['name']}' already exists, skipping...")
            else:
                print(f"Inserting boulder: {boulder['name']}")
                result = supabase.table("boulder").insert(boulder).execute()

        except Exception as e:
            print(f"Error inserting boulder '{boulder['name']}': {str(e)}")

    def display_boulder_data(self, boulder: Dict[str, Any]) -> None:
        """Display boulder data in a readable format."""
        print("\n==== BOULDER DATA ====")
        print(f"Name: {boulder['name']}")
        print(f"Grade: {boulder['grade']}")
        print(f"Description: {boulder['description']}")
        print(f"Location: Lat {boulder['latitude']}, Lon {boulder['longitude']}")
        print(f"Height: {boulder['height']}")
        print(f"Image URL: {boulder['image']}")
        print(f"Area ID: {boulder['areaId']}")
        print(f"Quality: {boulder['quality']}")
        print(f"Type: {boulder['type']}")
        print("=====================\n")

    async def import_boulders(self) -> None:
        """Import all boulders from the Excel file."""
        df = self.read_excel_file()

        print(f"\nFound {len(df)} boulders in Excel file. Preparing to import...\n")

        # Process each row
        for index, row in df.iterrows():
            try:
                boulder_data = self.prepare_boulder_data(row)

                # Display boulder data before insertion
                print(f"Row {index+1}/{len(df)}:")
                self.display_boulder_data(boulder_data)

                # Ask for confirmation before inserting (optional)
                # confirmation = input("Insert this boulder? (y/n): ").lower().strip()
                # if confirmation != 'y':
                #     print("Skipping this boulder.")
                #     continue

                # await self.insert_boulder(boulder_data)
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")

        print(f"Finished importing boulders from {self.excel_file_path}")

async def main():
    importer = BoulderImporter(BASE_PATH)
    await importer.import_boulders()

if __name__ == "__main__":
    asyncio.run(main())
