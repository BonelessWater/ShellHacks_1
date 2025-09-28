#!/usr/bin/env python3
"""
Script to pull PDF files from BigQuery database and download the first 5 invoices
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
from pathlib import Path

# Add the parent directory to import the BigQuery configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bigquery_config import bq_manager
    BIGQUERY_AVAILABLE = True
except ImportError as e:
    print(f"BigQuery not available: {e}")
    BIGQUERY_AVAILABLE = False
    bq_manager = None

class InvoicePDFDownloader:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.download_dir = Path("downloaded_invoices")
        self.download_dir.mkdir(exist_ok=True)
        
    async def get_invoices_from_db(self, limit=5):
        """Fetch invoices from BigQuery database"""
        if not BIGQUERY_AVAILABLE or not bq_manager:
            print("BigQuery not available, using API fallback...")
            return await self.get_invoices_from_api(limit)
        
        try:
            # Query to get invoices with potential PDF file references
            query = f"""
            SELECT 
                invoice_id,
                invoice_number,
                vendor_name,
                CAST(total_amount AS FLOAT64) as total_amount,
                invoice_date,
                verification_status,
                CAST(confidence_score AS FLOAT64) as confidence_score,
                processed_ts,
                source,
                invoice_hash
            FROM `vaulted-timing-473322-f9.document_forgery.invoices` 
            WHERE verification_status IS NOT NULL
            ORDER BY processed_ts DESC 
            LIMIT {limit}
            """
            
            df = bq_manager.query(query)
            
            if df.empty:
                print("No invoices found in database")
                return []
            
            # Convert DataFrame to list of dictionaries
            invoices = []
            for _, row in df.iterrows():
                invoice = {
                    'id': row['invoice_id'],
                    'invoice_number': row['invoice_number'],
                    'vendor_name': row['vendor_name'],
                    'total_amount': row['total_amount'],
                    'invoice_date': str(row['invoice_date']) if row['invoice_date'] else None,
                    'verification_status': row['verification_status'],
                    'confidence_score': row['confidence_score'],
                    'source': row['source'],
                    'invoice_hash': row['invoice_hash']
                }
                invoices.append(invoice)
            
            return invoices
            
        except Exception as e:
            print(f"Error querying BigQuery: {e}")
            print("Falling back to API...")
            return await self.get_invoices_from_api(limit)
    
    async def get_invoices_from_api(self, limit=5):
        """Fallback method to get invoices from the REST API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/api/invoices?limit={limit}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('invoices', [])
                    else:
                        print(f"API request failed with status {response.status}")
                        return []
        except Exception as e:
            print(f"Error fetching from API: {e}")
            return []
    
    async def download_invoice_pdf(self, session, invoice_data, index):
        """Download PDF for a single invoice"""
        invoice_id = invoice_data.get('id', f"invoice_{index}")
        invoice_number = invoice_data.get('invoice_number', f"INV-{index}")
        
        # Try multiple potential PDF download endpoints
        pdf_urls = [
            f"{self.api_base_url}/api/invoices/{invoice_id}/pdf",
            f"{self.api_base_url}/api/invoices/{invoice_id}/download",
            f"{self.api_base_url}/api/files/{invoice_id}.pdf"
        ]
        
        for url in pdf_urls:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'pdf' in content_type.lower():
                            # Save the PDF file
                            filename = f"{invoice_number}_{invoice_id}.pdf"
                            filepath = self.download_dir / filename
                            
                            async with aiofiles.open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            
                            print(f"✅ Downloaded: {filename} ({await self.get_file_size(filepath)} bytes)")
                            return True
                        else:
                            print(f"⚠️  URL {url} returned non-PDF content: {content_type}")
                    elif response.status == 404:
                        continue  # Try next URL
                    else:
                        print(f"⚠️  HTTP {response.status} for {url}")
            except Exception as e:
                print(f"⚠️  Error downloading from {url}: {e}")
                continue
        
        # If no PDF found, create a placeholder with invoice data
        await self.create_invoice_placeholder(invoice_data, index)
        return False
    
    async def create_invoice_placeholder(self, invoice_data, index):
        """Create a text file with invoice information if PDF not available"""
        invoice_id = invoice_data.get('id', f"invoice_{index}")
        invoice_number = invoice_data.get('invoice_number', f"INV-{index}")
        filename = f"{invoice_number}_{invoice_id}_info.txt"
        filepath = self.download_dir / filename
        
        invoice_info = f"""
INVOICE INFORMATION
==================
Invoice ID: {invoice_data.get('id', 'N/A')}
Invoice Number: {invoice_data.get('invoice_number', 'N/A')}
Vendor: {invoice_data.get('vendor_name', 'N/A')}
Amount: ${invoice_data.get('total_amount', 0):.2f}
Date: {invoice_data.get('invoice_date', 'N/A')}
Status: {invoice_data.get('verification_status', 'N/A')}
Confidence Score: {invoice_data.get('confidence_score', 'N/A')}
Source: {invoice_data.get('source', 'N/A')}

Note: PDF file not available for download. This is the extracted invoice data.
Generated: {datetime.now().isoformat()}
        """.strip()
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(invoice_info)
        
        print(f"📄 Created info file: {filename}")
    
    async def get_file_size(self, filepath):
        """Get file size in bytes"""
        try:
            stat = await aiofiles.os.stat(filepath)
            return stat.st_size
        except:
            return 0
    
    async def download_invoices(self, limit=5):
        """Main method to download invoice PDFs"""
        print(f"🔍 Fetching first {limit} invoices from database...")
        
        # Get invoice data from database
        invoices = await self.get_invoices_from_db(limit)
        
        if not invoices:
            print("❌ No invoices found")
            return
        
        print(f"📋 Found {len(invoices)} invoices")
        
        # Download PDFs
        async with aiohttp.ClientSession() as session:
            download_tasks = []
            for i, invoice in enumerate(invoices):
                task = self.download_invoice_pdf(session, invoice, i + 1)
                download_tasks.append(task)
            
            # Execute downloads concurrently
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Summary
        successful_downloads = sum(1 for r in results if r is True)
        print(f"\n📊 Download Summary:")
        print(f"   Total invoices: {len(invoices)}")
        print(f"   Successful PDF downloads: {successful_downloads}")
        print(f"   Info files created: {len(invoices) - successful_downloads}")
        print(f"   Download directory: {self.download_dir.absolute()}")

def main():
    """Main function to run the invoice PDF downloader"""
    print("🚀 Invoice PDF Downloader")
    print("=" * 50)
    
    downloader = InvoicePDFDownloader()
    
    # Run the async download process
    asyncio.run(downloader.download_invoices(limit=5))

if __name__ == "__main__":
    main()