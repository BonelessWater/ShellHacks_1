# backend/services/database.py
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to the path to import bigquery_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from bigquery_config import bq_manager
    BIGQUERY_AVAILABLE = True
except ImportError as e:
    print(f"BigQuery not available: {e}")
    BIGQUERY_AVAILABLE = False
    bq_manager = None

class DatabaseService:
    def __init__(self):
        self.bq_manager = bq_manager if BIGQUERY_AVAILABLE else None
        
    async def get_invoices(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch invoices from BigQuery database"""
        if not self.bq_manager:
            return self._get_mock_invoices(limit, offset)
        
        try:
            # First check if we can connect and if the table exists
            test_query = "SELECT COUNT(*) as total FROM `vaulted-timing-473322-f9.document_forgery.invoices` LIMIT 1"
            test_df = self.bq_manager.query(test_query)
            
            if test_df.empty:
                print("BigQuery table exists but is empty")
                return self._get_mock_invoices(limit, offset)
            
            # Query invoices from your BigQuery table
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
                vendor,
                line_items,
                subtotal,
                tax_amount,
                due_date,
                notes
            FROM `vaulted-timing-473322-f9.document_forgery.invoices` 
            ORDER BY processed_ts DESC 
            LIMIT {limit} OFFSET {offset}
            """
            
            df = self.bq_manager.query(query)
            
            # Convert DataFrame to the format expected by frontend
            invoices = []
            for _, row in df.iterrows():
                invoice = {
                    "id": row.get("invoice_id", ""),
                    "vendor": row.get("vendor_name", "Unknown Vendor"),
                    "amount": float(row.get("total_amount", 0)),
                    "status": self._map_verification_status(row.get("verification_status", "pending")),
                    "confidence": float(row.get("confidence_score", 0)),
                    "issues": self._calculate_issues(row),
                    "date": str(row.get("invoice_date", datetime.now().date())),
                    "description": f"Invoice from {row.get('vendor_name', 'Unknown')}"
                }
                invoices.append(invoice)
            
            # Get total count for pagination
            total = int(test_df.iloc[0]['total']) if not test_df.empty else len(invoices)
            
            return {
                "invoices": invoices,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(invoices)) < total
            }
            
        except Exception as e:
            print(f"BigQuery error: {e}")
            # Fallback to mock data for development
            return self._get_mock_invoices(limit, offset)
    
    def _get_mock_invoices(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Generate mock invoices data for development/fallback"""
        mock_invoices = [
            {
                "id": "INV-2025-0001",
                "vendor": "ACME Corp",
                "amount": 1050.00,
                "status": "approved",
                "confidence": 0.95,
                "issues": 0,
                "date": "2025-09-01",
                "description": "Monthly consulting services"
            },
            {
                "id": "INV-2025-0002", 
                "vendor": "TechSupply Inc",
                "amount": 2750.50,
                "status": "review_required",
                "confidence": 0.72,
                "issues": 2,
                "date": "2025-09-02",
                "description": "Hardware procurement"
            },
            {
                "id": "INV-2025-0003",
                "vendor": "Office Solutions LLC",
                "amount": 890.25,
                "status": "rejected",
                "confidence": 0.35,
                "issues": 5,
                "date": "2025-09-03", 
                "description": "Suspicious office supplies order"
            },
            {
                "id": "INV-2025-0004",
                "vendor": "CloudServices Pro",
                "amount": 5200.00,
                "status": "approved",
                "confidence": 0.88,
                "issues": 1,
                "date": "2025-09-04",
                "description": "Cloud infrastructure services"
            }
        ]
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_invoices = mock_invoices[start_idx:end_idx]
        
        return {
            "invoices": paginated_invoices,
            "total": len(mock_invoices),
            "limit": limit,
            "offset": offset,
            "has_more": end_idx < len(mock_invoices)
        }
    
    def _map_verification_status(self, status: str) -> str:
        """Map database status to frontend status"""
        status_mapping = {
            "pending": "review_required",
            "approved": "approved", 
            "rejected": "rejected",
            "flagged": "review_required",
            "verified": "approved"
        }
        return status_mapping.get(status.lower() if status else "", "review_required")
    
    def _calculate_issues(self, row) -> int:
        """Calculate number of issues based on confidence score and other factors"""
        confidence = float(row.get("confidence_score", 1.0))
        issues = 0
        
        if confidence < 0.5:
            issues += 3
        elif confidence < 0.7:
            issues += 2
        elif confidence < 0.9:
            issues += 1
            
        return issues
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status including database connection"""
        if not self.bq_manager:
            return {
                "status": "healthy",
                "agents_online": True,
                "database_connected": False,
                "processing_queue": 0,
                "system_details": {
                    "status": "operational",
                    "agents": ["vendor_check", "amount_check", "pattern_analysis"],
                    "database": "offline - using mock data"
                }
            }
        
        try:
            # Test database connection with a simple query
            test_query = "SELECT 1 as test LIMIT 1"
            test_df = self.bq_manager.query(test_query)
            
            if test_df.empty:
                raise Exception("Empty result from test query")
            
            # Try to get queue size (this might fail if table doesn't exist)
            processing_queue = 0
            try:
                queue_query = """
                SELECT COUNT(*) as pending_count 
                FROM `vaulted-timing-473322-f9.document_forgery.invoices` 
                WHERE verification_status = 'pending'
                LIMIT 1
                """
                queue_df = self.bq_manager.query(queue_query)
                processing_queue = int(queue_df.iloc[0]['pending_count']) if not queue_df.empty else 0
            except Exception:
                # Table might not exist, use default
                processing_queue = 0
            
            return {
                "status": "healthy",
                "agents_online": True,
                "database_connected": True,
                "processing_queue": processing_queue,
                "system_details": {
                    "status": "operational",
                    "agents": ["vendor_check", "amount_check", "pattern_analysis"],
                    "database": "connected - BigQuery"
                }
            }
            
        except Exception as e:
            print(f"BigQuery connection error: {e}")
            # Return operational status but indicate DB is offline
            return {
                "status": "healthy",
                "agents_online": True,
                "database_connected": False,
                "processing_queue": 0,
                "system_details": {
                    "status": "operational",
                    "agents": ["vendor_check", "amount_check", "pattern_analysis"],
                    "database": f"offline - {str(e)[:50]}..."
                }
            }
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get analytics data from the database"""
        if not self.bq_manager:
            return {"total_invoices": 0, "total_amount": 0, "average_confidence": 0}
        
        try:
            analytics_query = """
            SELECT 
                COUNT(*) as total_invoices,
                SUM(CAST(total_amount AS FLOAT64)) as total_amount,
                AVG(CAST(confidence_score AS FLOAT64)) as avg_confidence,
                COUNT(CASE WHEN verification_status = 'approved' THEN 1 END) as approved_count,
                COUNT(CASE WHEN verification_status = 'rejected' THEN 1 END) as rejected_count,
                COUNT(CASE WHEN verification_status = 'pending' THEN 1 END) as pending_count
            FROM `vaulted-timing-473322-f9.document_forgery.invoices`
            WHERE invoice_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            """
            
            df = self.bq_manager.query(analytics_query)
            
            if df.empty:
                return {"total_invoices": 0, "total_amount": 0, "average_confidence": 0}
            
            row = df.iloc[0]
            return {
                "total_invoices": int(row.get("total_invoices", 0)),
                "total_amount": float(row.get("total_amount", 0)),
                "approved_count": int(row.get("approved_count", 0)),
                "rejected_count": int(row.get("rejected_count", 0)),
                "pending_count": int(row.get("pending_count", 0)),
                "average_confidence": float(row.get("avg_confidence", 0)),
            }
            
        except Exception as e:
            print(f"Error fetching analytics: {e}")
            return {"total_invoices": 0, "total_amount": 0, "average_confidence": 0}

# Create singleton instance
database_service = DatabaseService()