#!/usr/bin/env python3
"""
Vercel Serverless Function for Invoice Fraud Detection API
Entry point for all API requests
"""

import json
import os
import sys
from typing import Any, Dict

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), "..")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from agents import create_agent_coordinator, validate_agent_config
    from main_pipeline import get_system_status, run_fraud_detection
except ImportError as e:
    print(f"Import error: {e}")

    # Fallback minimal implementation
    def run_fraud_detection(data, max_iterations=3):
        return {"error": "Backend not properly configured", "status": "error"}

    def get_system_status():
        return {"status": "error", "message": "Backend not available"}


def handler(request):
    """Main Vercel handler for all API requests"""

    # Parse request
    method = request.get("method", "GET")
    path = request.get("path", "/")

    # Remove /api prefix if present
    if path.startswith("/api"):
        path = path[4:]

    # Add CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Content-Type": "application/json",
    }

    try:
        # Handle OPTIONS for CORS preflight
        if method == "OPTIONS":
            return {"statusCode": 200, "headers": headers, "body": ""}

        # Route requests
        if path == "/health" or path == "/":
            result = health_check()

        elif path == "/system/status":
            result = get_system_status()

        elif path.startswith("/invoices/upload") and method == "POST":
            body = json.loads(request.get("body", "{}"))
            result = handle_invoice_upload(body)

        elif path.startswith("/invoices") and method == "GET":
            result = handle_get_invoices()

        elif path.startswith("/agents/config"):
            if method == "GET":
                result = handle_get_agent_config()
            elif method == "PUT":
                body = json.loads(request.get("body", "{}"))
                result = handle_update_agent_config(body)
            else:
                result = {"error": "Method not allowed", "status": "error"}

        else:
            result = {"error": "Endpoint not found", "status": "error"}

        return {"statusCode": 200, "headers": headers, "body": json.dumps(result)}

    except Exception as e:
        error_result = {"error": str(e), "status": "error", "type": "server_error"}

        return {"statusCode": 500, "headers": headers, "body": json.dumps(error_result)}


def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "Invoice Fraud Detection API",
        "version": "1.0.0",
        "timestamp": "2025-09-27",
    }


def handle_invoice_upload(data):
    """Handle invoice upload and fraud detection"""
    try:
        # Extract invoice data from request
        invoice_data = data.get("invoice_data", {})

        if not invoice_data:
            return {"error": "No invoice data provided", "status": "error"}

        # Run fraud detection
        result = run_fraud_detection(invoice_data, max_iterations=3)

        return {
            "status": "success",
            "result": result,
            "message": "Invoice processed successfully",
        }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "status": "error"}


def handle_get_invoices():
    """Handle get invoices request"""
    # Mock data for now - replace with actual database logic
    return {"status": "success", "invoices": [], "total": 0}


def handle_get_agent_config():
    """Get current agent configuration"""
    try:
        # Default configuration
        config = {
            "vendor_agent": {
                "enabled": True,
                "approved_vendors": ["Microsoft", "Amazon", "Google"],
            },
            "totals_agent": {"enabled": True, "max_amount": 10000},
            "pattern_agent": {
                "enabled": True,
                "suspicious_keywords": ["urgent", "wire transfer", "bitcoin"],
            },
        }

        return {"status": "success", "config": config}

    except Exception as e:
        return {"error": f"Failed to get config: {str(e)}", "status": "error"}


def handle_update_agent_config(data):
    """Update agent configuration"""
    try:
        config = data.get("config", {})

        # Validate config
        errors, warnings = validate_agent_config(config)

        if errors:
            return {
                "error": "Configuration validation failed",
                "errors": errors,
                "warnings": warnings,
                "status": "error",
            }

        # Save config (implement actual saving logic)
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "warnings": warnings,
        }

    except Exception as e:
        return {"error": f"Failed to update config: {str(e)}", "status": "error"}


# Vercel serverless function entry point
def lambda_handler(event, context):
    """AWS Lambda compatible handler"""
    return handler(event)


# For local testing
if __name__ == "__main__":
    # Test the handler locally
    test_request = {"method": "GET", "path": "/health"}

    result = handler(test_request)
    print(json.dumps(result, indent=2))
