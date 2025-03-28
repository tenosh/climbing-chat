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
from openpyxl import load_workbook

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configuration
BASE_PATH = "./climbs/san_luis_potosi/guadalcazar/boulders.xlsx"
SECTOR_ID = "5f08920b-ff8b-45ed-b3f8-a4976bdd71b7"
DEFAULT_DESCRIPTION = "Boulder en proceso de documentación. Si tienes información sobre este problema, ¡ayúdanos a completarla!"
DEFAULT_GRADE = "desconocido"
DEFAULT_QUALITY = 100
DEFAULT_TYPE = "boulder"
BUCKET_NAME = "cactux"
STORAGE_FOLDER = "boulders"

class BoulderImporter:
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path

    def read_excel_file(self) -> pd.DataFrame:
        """Read data from the Excel file including binary image data."""
        try:
            # First, load the workbook directly with openpyxl to access images
            wb = load_workbook(self.excel_file_path)
            sheet = wb.active

            # Create a dictionary to store image data by cell position
            image_data = {}

            # Extract images from the worksheet
            for image in sheet._images:
                # Get the cell position (row, col) where the image is anchored
                row = image.anchor._from.row + 1  # 1-based indexing
                col = image.anchor._from.col + 1  # 1-based indexing

                # Store the image data
                image_data[(row, col)] = image._data()

            # Now read the regular data with pandas
            df = pd.read_excel(self.excel_file_path, engine='openpyxl', header=None)

            # Add an image column if it doesn't exist
            if len(df.columns) <= 7:
                df['image'] = None

            # Determine the correct image column index
            image_col_name = df.columns[7] if len(df.columns) > 7 else 'image'

            # For each row in the dataframe, check if there's an image
            for i, row in df.iterrows():
                # Excel rows are 1-indexed, but pandas is 0-indexed
                excel_row = i + 1  # +1 because Excel is 1-indexed (removed the extra +1)

                # Check if there's an image for this cell
                if (excel_row, 8) in image_data:  # Column 8 in Excel (1-indexed)
                    # Add the image data to the dataframe
                    df.at[i, image_col_name] = image_data[(excel_row, 8)]

            return df
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def process_image(self, image_data: Any) -> Optional[str]:
        """
        Process an image from various sources:
        1. URL string
        2. Local file path string
        3. Binary data from Excel

        The image is:
        1. Cropped to 4:3 aspect ratio
        2. Optimized
        3. Converted to .webp format

        Returns the path to the processed image in Supabase storage
        """
        if image_data is None or pd.isna(image_data):
            print("No image data provided (None or NaN)")
            return None

        try:
            # Handle different types of image sources
            if isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    # Handle URL
                    print(f"Processing image from URL: {image_data}")
                    response = requests.get(image_data, stream=True)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content))
                else:
                    # Handle local file path
                    print(f"Processing image from local path: {image_data}")
                    img = Image.open(image_data)
            else:
                # Handle binary data from Excel
                print("Processing binary image data from Excel")
                img = Image.open(io.BytesIO(image_data))

            # Print image details after opening
            print(f"Successfully opened image: {img.format}, size: {img.size}, mode: {img.mode}")

            # Calculate dimensions for 4:3 aspect ratio crop
            width, height = img.size
            target_ratio = 3/4

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
            quality = 85  # Start with this quality
            max_size_kb = 300  # Maximum file size in KB

            # First, resize to max dimensions while maintaining aspect ratio
            max_width = 900
            max_height = 1200
            current_width, current_height = img.size

            # Calculate scaling factor to fit within max dimensions
            width_ratio = max_width / current_width
            height_ratio = max_height / current_height
            scale_factor = min(width_ratio, height_ratio)

            # Only resize if image is larger than the max dimensions
            if scale_factor < 1:
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"Resized image to {new_width}x{new_height} to fit within max dimensions")

            current_width, current_height = img.size

            while True:
                img.save(temp_path, format="WEBP", quality=quality, optimize=True)
                file_size_kb = os.path.getsize(temp_path) / 1024

                if file_size_kb <= max_size_kb:
                    print(f"Saved image at quality {quality}, size: {file_size_kb:.2f}KB, dimensions: {img.size}")
                    break

                if quality > 10:
                    # First try reducing quality
                    quality -= 10
                    print(f"Image too large ({file_size_kb:.2f}KB), reducing quality to {quality}")
                else:
                    # If quality is already at minimum, reduce dimensions
                    new_width = int(current_width * 0.8)
                    new_height = int(current_height * 0.8)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    current_width, current_height = new_width, new_height
                    quality = 60  # Reset quality after resizing
                    print(f"Image too large at minimum quality, resizing to {new_width}x{new_height}")

                    # If image becomes too small, stop resizing
                    if new_width < 500 or new_height < 500:
                        print("Warning: Image dimensions getting too small, saving anyway")
                        img.save(temp_path, format="WEBP", quality=quality, optimize=True)
                        break

            print(f"Saved temporary image to {temp_path}")

            # Upload to Supabase storage
            storage_path = f"{STORAGE_FOLDER}/{unique_id}.webp"
            with open(temp_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "image/webp"}
                )
            print(f"Uploaded image to Supabase storage: {storage_path}")

            # Clean up temp file
            os.remove(temp_path)

            # Return the public URL
            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)

            # Remove trailing question mark if present
            if public_url.endswith('?'):
                public_url = public_url[:-1]

            print(f"Generated public URL: {public_url}")
            return public_url

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            return None

    def prepare_boulder_data(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a row of Excel data to a boulder database record."""
        # Process image if available - the image is in the 8th column (index 7)
        image_data = None
        if len(row) > 7:
            image_data = row.iloc[7]

        print(f"\nProcessing image for boulder: {row.iloc[0]}")
        image_url = self.process_image(image_data)

        # Convert coordinates from DMS to decimal degrees if needed
        lat = row.iloc[3] if len(row) > 3 else None
        lon = row.iloc[4] if len(row) > 4 else None

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
            "name": row.iloc[0],  # Column 1 (index 0) is the name
            "grade": row.iloc[1] if pd.notna(row.iloc[1]) else DEFAULT_GRADE,  # Use default grade if none provided
            "description": row.iloc[2] if pd.notna(row.iloc[2]) else DEFAULT_DESCRIPTION,  # Column 3 (index 2) is the description
            "latitude": lat,  # Using converted latitude
            "longitude": lon,  # Using converted longitude
            "height": row.iloc[5] if len(row) > 5 and pd.notna(row.iloc[5]) else None,  # Column 6 (index 5) is height
            "quality": self.extract_quality(row),  # Extract quality from the document
            "image": image_url,
            "sector_id": SECTOR_ID,
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

    def extract_quality(self, row: pd.Series) -> int:
        """Extract quality from the row, converting to int if necessary."""
        # Quality is in column 7 (index 6)
        if len(row) > 6 and pd.notna(row.iloc[6]):
            try:
                return int(row.iloc[6])
            except (ValueError, TypeError):
                # If conversion fails, use default
                print(f"Could not convert quality '{row.iloc[6]}' to integer, using default")
                return DEFAULT_QUALITY

        # If no quality found or it's NaN, use default
        return DEFAULT_QUALITY

    async def insert_boulder(self, boulder: Dict[str, Any]) -> None:
        """Insert a boulder into the database."""
        try:
            # Check if boulder already exists with the same name in this area
            existing = supabase.table("boulder").select("id").eq("sector_id", SECTOR_ID).eq("name", boulder["name"]).execute()

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
        print(f"Quality: {boulder['quality']}")  # Moved quality before image
        print(f"Image: {boulder['image']}")
        print(f"Sector ID: {boulder['sector_id']}")
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

                await self.insert_boulder(boulder_data)
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")

        print(f"Finished importing boulders from {self.excel_file_path}")

async def main():
    importer = BoulderImporter(BASE_PATH)
    await importer.import_boulders()

if __name__ == "__main__":
    asyncio.run(main())
