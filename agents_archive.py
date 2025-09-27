#!/usr/bin/env python3
"""
Specialized Invoice Verification Agents System
Each agent is specialized for specific invoice verification tasks
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import uuid
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from collections import defaultdict, Counter
import math

# Configure comprehensive logging
class InvoiceSystemLogger:
    """Centralized logging system for invoice verification agents"""

    def __init__(self, log_dir: str = "invoice_verification_system/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create separate loggers for each specialized agent
        self.agents = [
            'orchestrator', 'data_extraction', 'duplicate_detection',
            'amount_validation', 'vendor_verification', 'date_validation',
            'tax_calculation', 'po_matching', 'fraud_detection',
            'compliance_check', 'analytics_engine', 'groupchat'
        ]

        self.loggers = {}
        self._setup_loggers()

    def _setup_loggers(self):
        """Setup loggers for all specialized agents"""
        for agent_name in self.agents:
            logger = logging.getLogger(f"invoice_{agent_name}")
            logger.setLevel(logging.DEBUG)

            # File handler
            file_handler = logging.FileHandler(self.log_dir / f"{agent_name}.log")
            file_handler.setLevel(logging.DEBUG)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            self.loggers[agent_name] = logger

    def get_logger(self, agent_name: str) -> logging.Logger:
        """Get logger for specific agent"""
        return self.loggers.get(agent_name, logging.getLogger(f"invoice_{agent_name}"))

# Initialize global logger
invoice_logger = InvoiceSystemLogger()

@dataclass
class InvoiceData:
    """Standard invoice data structure based on provided samples"""
    invoice_number: str
    vendor: Dict[str, str]  # name, address, phone, email, tax_id
    invoice_date: str
    due_date: str
    subtotal: float
    tax_amount: float
    total_amount: float
    payment_terms: str
    purchase_order: Optional[str]
    line_items: List[Dict[str, Any]]  # description, quantity, unit_price, total
    notes: str

    # Additional fields for verification
    received_date: Optional[str] = None
    processed_date: Optional[str] = None
    verification_status: Optional[str] = None
    confidence_score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvoiceData':
        """Create InvoiceData from dictionary"""
        return cls(
            invoice_number=data.get('invoice_number', ''),
            vendor=data.get('vendor', {}),
            invoice_date=data.get('invoice_date', ''),
            due_date=data.get('due_date', ''),
            subtotal=float(data.get('subtotal', 0)),
            tax_amount=float(data.get('tax_amount', 0)),
            total_amount=float(data.get('total_amount', 0)),
            payment_terms=data.get('payment_terms', ''),
            purchase_order=data.get('purchase_order'),
            line_items=data.get('line_items', []),
            notes=data.get('notes', ''),
            received_date=data.get('received_date'),
            processed_date=data.get('processed_date'),
            verification_status=data.get('verification_status'),
            confidence_score=data.get('confidence_score')
        )

@dataclass
class VerificationResult:
    """Result from specialized verification agent"""
    agent_name: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    confidence: float  # 0.0 to 1.0
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DatabaseManager:
    """Manages invoice database for duplicate detection and analytics"""

    def __init__(self, db_path: str = "invoice_verification_system/data/invoices.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_number TEXT UNIQUE,
                    vendor_name TEXT,
                    vendor_tax_id TEXT,
                    invoice_date TEXT,
                    due_date TEXT,
                    subtotal REAL,
                    tax_amount REAL,
                    total_amount REAL,
                    payment_terms TEXT,
                    purchase_order TEXT,
                    line_items_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verification_status TEXT,
                    confidence_score REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_number TEXT,
                    agent_name TEXT,
                    status TEXT,
                    confidence REAL,
                    findings TEXT,
                    recommendations TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def save_invoice(self, invoice: InvoiceData) -> bool:
        """Save invoice to database"""
        try:
            line_items_hash = hashlib.md5(
                json.dumps(invoice.line_items, sort_keys=True).encode()
            ).hexdigest()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO invoices
                    (invoice_number, vendor_name, vendor_tax_id, invoice_date, due_date,
                     subtotal, tax_amount, total_amount, payment_terms, purchase_order,
                     line_items_hash, verification_status, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    invoice.invoice_number,
                    invoice.vendor.get('name', ''),
                    invoice.vendor.get('tax_id', ''),
                    invoice.invoice_date,
                    invoice.due_date,
                    invoice.subtotal,
                    invoice.tax_amount,
                    invoice.total_amount,
                    invoice.payment_terms,
                    invoice.purchase_order,
                    line_items_hash,
                    invoice.verification_status,
                    invoice.confidence_score
                ))
                conn.commit()
            return True
        except Exception as e:
            invoice_logger.get_logger('data_extraction').error(f"Error saving invoice: {e}")
            return False

    def find_duplicates(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Find potential duplicate invoices"""
        try:
            line_items_hash = hashlib.md5(
                json.dumps(invoice.line_items, sort_keys=True).encode()
            ).hexdigest()

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Check for exact matches
                exact_matches = conn.execute("""
                    SELECT * FROM invoices
                    WHERE invoice_number = ? AND vendor_name = ?
                """, (invoice.invoice_number, invoice.vendor.get('name', ''))).fetchall()

                # Check for amount and vendor matches
                amount_matches = conn.execute("""
                    SELECT * FROM invoices
                    WHERE vendor_name = ? AND total_amount = ? AND invoice_number != ?
                """, (
                    invoice.vendor.get('name', ''),
                    invoice.total_amount,
                    invoice.invoice_number
                )).fetchall()

                # Check for line items hash matches
                hash_matches = conn.execute("""
                    SELECT * FROM invoices
                    WHERE line_items_hash = ? AND invoice_number != ?
                """, (line_items_hash, invoice.invoice_number)).fetchall()

                duplicates = []
                for match in exact_matches:
                    duplicates.append({
                        'type': 'exact_match',
                        'severity': 'critical',
                        'existing_invoice': dict(match),
                        'confidence': 1.0
                    })

                for match in amount_matches:
                    duplicates.append({
                        'type': 'amount_vendor_match',
                        'severity': 'high',
                        'existing_invoice': dict(match),
                        'confidence': 0.8
                    })

                for match in hash_matches:
                    duplicates.append({
                        'type': 'line_items_match',
                        'severity': 'high',
                        'existing_invoice': dict(match),
                        'confidence': 0.9
                    })

                return duplicates

        except Exception as e:
            invoice_logger.get_logger('duplicate_detection').error(f"Error finding duplicates: {e}")
            return []

    def get_vendor_history(self, vendor_name: str) -> Dict[str, Any]:
        """Get vendor transaction history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                summary = conn.execute("""
                    SELECT COUNT(*) as total_invoices,
                           AVG(total_amount) as avg_amount,
                           SUM(total_amount) as total_amount,
                           AVG(confidence_score) as avg_confidence,
                           MIN(created_at) as first_transaction,
                           MAX(created_at) as last_transaction
                    FROM invoices
                    WHERE vendor_name = ?
                """, (vendor_name,)).fetchone()

                recent = conn.execute("""
                    SELECT * FROM invoices
                    WHERE vendor_name = ?
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (vendor_name,)).fetchall()

                return {
                    'summary': dict(summary) if summary else {},
                    'recent_invoices': [dict(row) for row in recent]
                }

        except Exception as e:
            invoice_logger.get_logger('vendor_verification').error(f"Error getting vendor history: {e}")
            return {'summary': {}, 'recent_invoices': []}

# Initialize database
db_manager = DatabaseManager()

class SpecializedAgent:
    """Base class for specialized invoice verification agents"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = invoice_logger.get_logger(name.lower().replace(' ', '_'))
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'success_rate': 0.0
        }

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Process invoice - to be implemented by subclasses"""
        raise NotImplementedError

    def _create_result(self, status: str, confidence: float, findings: List[Dict],
                      recommendations: List[str], processing_time: float) -> VerificationResult:
        """Create standardized verification result"""
        return VerificationResult(
            agent_name=self.name,
            status=status,
            confidence=confidence,
            findings=findings,
            recommendations=recommendations,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

class DataExtractionAgent(SpecializedAgent):
    """Specialized agent for extracting and validating invoice data structure"""

    def __init__(self):
        super().__init__(
            name="Data Extraction Agent",
            description="Extracts and validates invoice data structure and completeness"
        )

        self.required_fields = [
            'invoice_number', 'vendor', 'invoice_date', 'total_amount', 'line_items'
        ]

        self.vendor_required_fields = [
            'name', 'address', 'phone', 'email', 'tax_id'
        ]

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Extract and validate invoice data structure"""
        start_time = datetime.now()
        self.logger.info(f"Processing data extraction for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # Check required fields
        missing_fields = self._check_required_fields(invoice)
        if missing_fields:
            findings.append({
                'type': 'missing_fields',
                'severity': 'high',
                'description': f"Missing required fields: {', '.join(missing_fields)}",
                'fields': missing_fields
            })
            confidence -= 0.3
            recommendations.append(f"Obtain missing fields: {', '.join(missing_fields)}")

        # Validate vendor information
        vendor_issues = self._validate_vendor_info(invoice.vendor)
        if vendor_issues:
            findings.extend(vendor_issues)
            confidence -= 0.2
            recommendations.append("Complete vendor information verification")

        # Validate line items structure
        line_item_issues = self._validate_line_items(invoice.line_items)
        if line_item_issues:
            findings.extend(line_item_issues)
            confidence -= 0.1
            recommendations.append("Review and correct line item calculations")

        # Data quality assessment
        quality_score = self._assess_data_quality(invoice)
        if quality_score < 0.8:
            findings.append({
                'type': 'data_quality',
                'severity': 'medium',
                'description': f"Data quality score: {quality_score:.2f}",
                'score': quality_score
            })
            recommendations.append("Improve data quality and completeness")

        # Advanced data extraction features
        extracted_data = await self._extract_advanced_features(invoice)
        if extracted_data['insights']:
            findings.append({
                'type': 'extracted_insights',
                'severity': 'info',
                'description': "Additional data insights extracted",
                'insights': extracted_data['insights']
            })

        # Save to database
        db_manager.save_invoice(invoice)

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _check_required_fields(self, invoice: InvoiceData) -> List[str]:
        """Check for missing required fields"""
        missing = []

        if not invoice.invoice_number:
            missing.append('invoice_number')
        if not invoice.vendor:
            missing.append('vendor')
        if not invoice.invoice_date:
            missing.append('invoice_date')
        if not invoice.total_amount:
            missing.append('total_amount')
        if not invoice.line_items:
            missing.append('line_items')

        return missing

    def _validate_vendor_info(self, vendor: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate vendor information completeness"""
        issues = []

        for field in self.vendor_required_fields:
            if not vendor.get(field):
                issues.append({
                    'type': 'incomplete_vendor_info',
                    'severity': 'medium',
                    'description': f"Missing vendor {field}",
                    'field': field
                })

        # Validate email format
        email = vendor.get('email', '')
        if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            issues.append({
                'type': 'invalid_email',
                'severity': 'low',
                'description': f"Invalid email format: {email}",
                'value': email
            })

        # Validate tax ID format (basic check)
        tax_id = vendor.get('tax_id', '')
        if tax_id and not re.match(r'^\d{2}-\d{7}$', tax_id):
            issues.append({
                'type': 'invalid_tax_id',
                'severity': 'medium',
                'description': f"Invalid tax ID format: {tax_id}",
                'value': tax_id
            })

        return issues

    def _validate_line_items(self, line_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate line items structure and calculations"""
        issues = []

        for i, item in enumerate(line_items):
            # Check required fields
            if not item.get('description'):
                issues.append({
                    'type': 'missing_line_item_description',
                    'severity': 'medium',
                    'description': f"Line item {i+1} missing description",
                    'line_item': i+1
                })

            # Validate calculations
            quantity = float(item.get('quantity', 0))
            unit_price = float(item.get('unit_price', 0))
            total = float(item.get('total', 0))

            expected_total = quantity * unit_price
            if abs(total - expected_total) > 0.01:  # Allow for rounding
                issues.append({
                    'type': 'line_item_calculation_error',
                    'severity': 'high',
                    'description': f"Line item {i+1} calculation error: {total} != {expected_total}",
                    'line_item': i+1,
                    'expected': expected_total,
                    'actual': total
                })

        return issues

    def _assess_data_quality(self, invoice: InvoiceData) -> float:
        """Assess overall data quality score"""
        score = 1.0

        # Completeness check
        total_fields = 11
        completed_fields = 0

        if invoice.invoice_number: completed_fields += 1
        if invoice.vendor: completed_fields += 1
        if invoice.invoice_date: completed_fields += 1
        if invoice.due_date: completed_fields += 1
        if invoice.subtotal: completed_fields += 1
        if invoice.tax_amount: completed_fields += 1
        if invoice.total_amount: completed_fields += 1
        if invoice.payment_terms: completed_fields += 1
        if invoice.purchase_order: completed_fields += 1
        if invoice.line_items: completed_fields += 1
        if invoice.notes: completed_fields += 1

        completeness_score = completed_fields / total_fields

        # Consistency check
        consistency_score = 1.0
        if invoice.subtotal and invoice.tax_amount and invoice.total_amount:
            expected_total = invoice.subtotal + invoice.tax_amount
            if abs(invoice.total_amount - expected_total) > 0.01:
                consistency_score -= 0.2

        return (completeness_score + consistency_score) / 2

    async def _extract_advanced_features(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Extract advanced features from invoice data"""
        insights = []

        # Pattern detection
        if len(invoice.line_items) == 1:
            insights.append("Single line item invoice - may be service-based")
        elif len(invoice.line_items) > 10:
            insights.append("High line item count - bulk purchase invoice")

        # Amount analysis
        if invoice.total_amount > 10000:
            insights.append("High-value invoice requiring additional approval")

        # Vendor analysis
        vendor_name = invoice.vendor.get('name', '').lower()
        if any(keyword in vendor_name for keyword in ['urgent', 'rush', 'emergency']):
            insights.append("Vendor name suggests urgent processing")

        return {'insights': insights}

class DuplicateDetectionAgent(SpecializedAgent):
    """Specialized agent for detecting duplicate invoices"""

    def __init__(self):
        super().__init__(
            name="Duplicate Detection Agent",
            description="Detects potential duplicate invoices using multiple criteria"
        )
        self.similarity_threshold = 0.85

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Detect duplicate invoices"""
        start_time = datetime.now()
        self.logger.info(f"Processing duplicate detection for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # Find potential duplicates
        duplicates = db_manager.find_duplicates(invoice)

        if duplicates:
            for duplicate in duplicates:
                findings.append({
                    'type': 'potential_duplicate',
                    'severity': duplicate['severity'],
                    'description': f"Potential duplicate found: {duplicate['type']}",
                    'duplicate_info': duplicate,
                    'confidence': duplicate['confidence']
                })

                if duplicate['severity'] == 'critical':
                    confidence = 0.0
                    recommendations.append("Reject invoice - exact duplicate detected")
                elif duplicate['severity'] == 'high':
                    confidence -= 0.4
                    recommendations.append("Review for potential duplicate")

        # Advanced similarity detection
        similar_invoices = await self._find_similar_invoices(invoice)
        if similar_invoices:
            findings.extend(similar_invoices)
            confidence -= 0.2
            recommendations.append("Investigate similar invoices")

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.3 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    async def _find_similar_invoices(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Find invoices with high similarity scores"""
        similar = []

        try:
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Find invoices with similar amounts (within 5%)
                amount_range = invoice.total_amount * 0.05
                similar_amounts = conn.execute("""
                    SELECT * FROM invoices
                    WHERE total_amount BETWEEN ? AND ?
                    AND vendor_name = ?
                    AND invoice_number != ?
                """, (
                    invoice.total_amount - amount_range,
                    invoice.total_amount + amount_range,
                    invoice.vendor.get('name', ''),
                    invoice.invoice_number
                )).fetchall()

                for similar_invoice in similar_amounts:
                    similarity_score = self._calculate_similarity(invoice, dict(similar_invoice))
                    if similarity_score > self.similarity_threshold:
                        similar.append({
                            'type': 'similar_invoice',
                            'severity': 'medium',
                            'description': f"Similar invoice found (similarity: {similarity_score:.2f})",
                            'similar_invoice': dict(similar_invoice),
                            'similarity_score': similarity_score
                        })

        except Exception as e:
            self.logger.error(f"Error finding similar invoices: {e}")

        return similar

    def _calculate_similarity(self, invoice1: InvoiceData, invoice2: Dict[str, Any]) -> float:
        """Calculate similarity score between two invoices"""
        score = 0.0

        # Amount similarity (40% weight)
        amount_diff = abs(invoice1.total_amount - invoice2.get('total_amount', 0))
        max_amount = max(invoice1.total_amount, invoice2.get('total_amount', 1))
        amount_similarity = 1 - (amount_diff / max_amount)
        score += amount_similarity * 0.4

        # Date similarity (30% weight)
        try:
            date1 = datetime.fromisoformat(invoice1.invoice_date)
            date2 = datetime.fromisoformat(invoice2.get('invoice_date', ''))
            date_diff = abs((date1 - date2).days)
            date_similarity = max(0, 1 - (date_diff / 30))  # 30 days max difference
            score += date_similarity * 0.3
        except:
            pass

        # Vendor similarity (20% weight)
        vendor1 = invoice1.vendor.get('name', '').lower()
        vendor2 = invoice2.get('vendor_name', '').lower()
        if vendor1 == vendor2:
            score += 0.2

        # Payment terms similarity (10% weight)
        if invoice1.payment_terms == invoice2.get('payment_terms', ''):
            score += 0.1

        return score

class AmountValidationAgent(SpecializedAgent):
    """Specialized agent for validating invoice amounts and calculations"""

    def __init__(self):
        super().__init__(
            name="Amount Validation Agent",
            description="Validates invoice amounts, calculations, and mathematical consistency"
        )
        self.tolerance = 0.01  # $0.01 tolerance for rounding

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Validate invoice amounts and calculations"""
        start_time = datetime.now()
        self.logger.info(f"Processing amount validation for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # Validate subtotal calculation
        subtotal_issues = self._validate_subtotal(invoice)
        findings.extend(subtotal_issues)
        if subtotal_issues:
            confidence -= 0.3
            recommendations.append("Verify subtotal calculation against line items")

        # Validate tax calculation
        tax_issues = self._validate_tax_calculation(invoice)
        findings.extend(tax_issues)
        if tax_issues:
            confidence -= 0.2
            recommendations.append("Review tax calculation")

        # Validate total calculation
        total_issues = self._validate_total(invoice)
        findings.extend(total_issues)
        if total_issues:
            confidence -= 0.4
            recommendations.append("Correct total amount calculation")

        # Check for unusual amounts
        unusual_amount_issues = self._check_unusual_amounts(invoice)
        findings.extend(unusual_amount_issues)
        if unusual_amount_issues:
            confidence -= 0.1
            recommendations.append("Review unusual amount patterns")

        # Validate line item calculations
        line_item_issues = self._validate_line_item_amounts(invoice)
        findings.extend(line_item_issues)
        if line_item_issues:
            confidence -= 0.2
            recommendations.append("Verify line item calculations")

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _validate_subtotal(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate subtotal against line items"""
        issues = []

        if not invoice.line_items:
            return issues

        calculated_subtotal = sum(
            float(item.get('total', 0)) for item in invoice.line_items
        )

        difference = abs(invoice.subtotal - calculated_subtotal)
        if difference > self.tolerance:
            issues.append({
                'type': 'subtotal_mismatch',
                'severity': 'high',
                'description': f"Subtotal mismatch: {invoice.subtotal} vs calculated {calculated_subtotal}",
                'expected': calculated_subtotal,
                'actual': invoice.subtotal,
                'difference': difference
            })

        return issues

    def _validate_tax_calculation(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate tax calculation"""
        issues = []

        if invoice.subtotal <= 0:
            return issues

        tax_rate = invoice.tax_amount / invoice.subtotal if invoice.subtotal > 0 else 0

        # Check for reasonable tax rates (0% to 15%)
        if tax_rate > 0.15:
            issues.append({
                'type': 'high_tax_rate',
                'severity': 'medium',
                'description': f"Tax rate {tax_rate*100:.2f}% exceeds typical range",
                'tax_rate': tax_rate,
                'tax_amount': invoice.tax_amount,
                'subtotal': invoice.subtotal
            })

        # Check for exact percentages (might indicate manual calculation)
        if tax_rate > 0:
            rate_percentage = tax_rate * 100
            if rate_percentage == round(rate_percentage):
                issues.append({
                    'type': 'exact_tax_percentage',
                    'severity': 'low',
                    'description': f"Tax rate is exact percentage: {rate_percentage:.0f}%",
                    'tax_rate': tax_rate
                })

        return issues

    def _validate_total(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate total amount calculation"""
        issues = []

        expected_total = invoice.subtotal + invoice.tax_amount
        difference = abs(invoice.total_amount - expected_total)

        if difference > self.tolerance:
            issues.append({
                'type': 'total_mismatch',
                'severity': 'critical',
                'description': f"Total mismatch: {invoice.total_amount} vs calculated {expected_total}",
                'expected': expected_total,
                'actual': invoice.total_amount,
                'difference': difference
            })

        return issues

    def _check_unusual_amounts(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Check for unusual amount patterns"""
        issues = []

        # Check for round numbers (might indicate estimates)
        if invoice.total_amount == round(invoice.total_amount) and invoice.total_amount > 100:
            issues.append({
                'type': 'round_amount',
                'severity': 'low',
                'description': f"Total amount is round number: ${invoice.total_amount:.0f}",
                'amount': invoice.total_amount
            })

        # Check for very small amounts
        if invoice.total_amount < 1.0:
            issues.append({
                'type': 'very_small_amount',
                'severity': 'medium',
                'description': f"Unusually small total: ${invoice.total_amount:.2f}",
                'amount': invoice.total_amount
            })

        # Check for very large amounts
        if invoice.total_amount > 100000:
            issues.append({
                'type': 'very_large_amount',
                'severity': 'medium',
                'description': f"Unusually large total: ${invoice.total_amount:.2f}",
                'amount': invoice.total_amount
            })

        return issues

    def _validate_line_item_amounts(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate individual line item calculations"""
        issues = []

        for i, item in enumerate(invoice.line_items):
            try:
                quantity = float(item.get('quantity', 0))
                unit_price = float(item.get('unit_price', 0))
                total = float(item.get('total', 0))

                expected_total = quantity * unit_price
                difference = abs(total - expected_total)

                if difference > self.tolerance:
                    issues.append({
                        'type': 'line_item_calculation_error',
                        'severity': 'high',
                        'description': f"Line item {i+1}: {total} != {quantity} × {unit_price}",
                        'line_item': i+1,
                        'quantity': quantity,
                        'unit_price': unit_price,
                        'expected': expected_total,
                        'actual': total,
                        'difference': difference
                    })
            except (ValueError, TypeError):
                issues.append({
                    'type': 'invalid_line_item_amounts',
                    'severity': 'high',
                    'description': f"Line item {i+1} has invalid numeric values",
                    'line_item': i+1
                })

        return issues

class VendorVerificationAgent(SpecializedAgent):
    """Specialized agent for vendor verification and validation"""

    def __init__(self):
        super().__init__(
            name="Vendor Verification Agent",
            description="Verifies vendor information and maintains vendor history"
        )
        # Mock vendor whitelist - in real implementation, this would be a database
        self.approved_vendors = {
            'ABC Office Supplies': {'tax_id': '12-3456789', 'status': 'approved'},
            'TechCorp Solutions': {'tax_id': '98-7654321', 'status': 'approved'},
            'Global Services Inc': {'tax_id': '45-6789012', 'status': 'pending'}
        }

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Verify vendor information"""
        start_time = datetime.now()
        self.logger.info(f"Processing vendor verification for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        vendor_name = invoice.vendor.get('name', '')

        # Vendor whitelist check
        whitelist_issues = self._check_vendor_whitelist(vendor_name, invoice.vendor)
        findings.extend(whitelist_issues)
        if whitelist_issues:
            confidence -= 0.3
            recommendations.append("Verify vendor approval status")

        # Vendor information validation
        info_issues = self._validate_vendor_information(invoice.vendor)
        findings.extend(info_issues)
        if info_issues:
            confidence -= 0.2
            recommendations.append("Complete vendor information")

        # Historical analysis
        history_analysis = await self._analyze_vendor_history(vendor_name)
        if history_analysis['issues']:
            findings.extend(history_analysis['issues'])
            confidence -= 0.1
            recommendations.append("Review vendor transaction history")

        # Add vendor insights
        if history_analysis['insights']:
            findings.append({
                'type': 'vendor_insights',
                'severity': 'info',
                'description': "Vendor historical insights",
                'insights': history_analysis['insights']
            })

        # Risk assessment
        risk_assessment = self._assess_vendor_risk(vendor_name, invoice, history_analysis)
        findings.append({
            'type': 'vendor_risk_assessment',
            'severity': 'info',
            'description': f"Vendor risk level: {risk_assessment['level']}",
            'risk_assessment': risk_assessment
        })

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _check_vendor_whitelist(self, vendor_name: str, vendor_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """Check vendor against approved list"""
        issues = []

        if vendor_name not in self.approved_vendors:
            issues.append({
                'type': 'vendor_not_approved',
                'severity': 'high',
                'description': f"Vendor '{vendor_name}' not in approved vendor list",
                'vendor_name': vendor_name
            })
        else:
            approved_info = self.approved_vendors[vendor_name]
            vendor_tax_id = vendor_info.get('tax_id', '')

            if vendor_tax_id != approved_info['tax_id']:
                issues.append({
                    'type': 'tax_id_mismatch',
                    'severity': 'critical',
                    'description': f"Tax ID mismatch for {vendor_name}",
                    'expected': approved_info['tax_id'],
                    'actual': vendor_tax_id
                })

            if approved_info['status'] != 'approved':
                issues.append({
                    'type': 'vendor_not_fully_approved',
                    'severity': 'medium',
                    'description': f"Vendor status: {approved_info['status']}",
                    'status': approved_info['status']
                })

        return issues

    def _validate_vendor_information(self, vendor: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate vendor information completeness and format"""
        issues = []

        # Required fields check
        required_fields = ['name', 'address', 'tax_id']
        for field in required_fields:
            if not vendor.get(field):
                issues.append({
                    'type': 'missing_vendor_field',
                    'severity': 'medium',
                    'description': f"Missing vendor {field}",
                    'field': field
                })

        # Validate email format
        email = vendor.get('email', '')
        if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            issues.append({
                'type': 'invalid_vendor_email',
                'severity': 'low',
                'description': f"Invalid vendor email format: {email}",
                'email': email
            })

        # Validate phone format
        phone = vendor.get('phone', '')
        if phone and not re.match(r'^\+?[\d\s\-\(\)]+$', phone):
            issues.append({
                'type': 'invalid_vendor_phone',
                'severity': 'low',
                'description': f"Invalid vendor phone format: {phone}",
                'phone': phone
            })

        return issues

    async def _analyze_vendor_history(self, vendor_name: str) -> Dict[str, Any]:
        """Analyze vendor transaction history"""
        history = db_manager.get_vendor_history(vendor_name)
        issues = []
        insights = []

        summary = history.get('summary', {})
        total_invoices = summary.get('total_invoices', 0)
        avg_confidence = summary.get('avg_confidence', 0)

        if total_invoices == 0:
            insights.append("New vendor - no transaction history")
        else:
            insights.append(f"Historical transactions: {total_invoices}")

            if avg_confidence is not None and avg_confidence < 0.7:
                issues.append({
                    'type': 'low_historical_confidence',
                    'severity': 'medium',
                    'description': f"Low average confidence in past transactions: {avg_confidence:.2f}",
                    'avg_confidence': avg_confidence
                })

            if total_invoices < 5:
                insights.append("Limited transaction history")
            elif total_invoices > 100:
                insights.append("Established vendor with extensive history")

        return {'issues': issues, 'insights': insights, 'summary': summary}

    def _assess_vendor_risk(self, vendor_name: str, invoice: InvoiceData,
                           history_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess vendor risk level"""
        risk_score = 0.0
        risk_factors = []

        # New vendor risk
        total_invoices = history_analysis['summary'].get('total_invoices', 0)
        if total_invoices == 0:
            risk_score += 0.3
            risk_factors.append("New vendor")
        elif total_invoices < 5:
            risk_score += 0.1
            risk_factors.append("Limited history")

        # Amount risk
        if invoice.total_amount > 10000:
            risk_score += 0.2
            risk_factors.append("High invoice amount")

        # Confidence risk
        avg_confidence = history_analysis['summary'].get('avg_confidence', 1.0) or 1.0
        if avg_confidence < 0.7:
            risk_score += 0.3
            risk_factors.append("Low historical confidence")

        # Approval status risk
        if vendor_name not in self.approved_vendors:
            risk_score += 0.4
            risk_factors.append("Not in approved vendor list")

        risk_level = 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'

        return {
            'score': min(1.0, risk_score),
            'level': risk_level,
            'factors': risk_factors
        }

class DateValidationAgent(SpecializedAgent):
    """Specialized agent for validating invoice dates and business rules"""

    def __init__(self):
        super().__init__(
            name="Date Validation Agent",
            description="Validates invoice dates and applies business date rules"
        )
        self.max_future_days = 30
        self.max_past_days = 365

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Validate invoice dates"""
        start_time = datetime.now()
        self.logger.info(f"Processing date validation for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # Validate invoice date
        invoice_date_issues = self._validate_invoice_date(invoice.invoice_date)
        findings.extend(invoice_date_issues)
        if invoice_date_issues:
            confidence -= 0.3
            recommendations.append("Review invoice date validity")

        # Validate due date
        due_date_issues = self._validate_due_date(invoice.due_date, invoice.invoice_date)
        findings.extend(due_date_issues)
        if due_date_issues:
            confidence -= 0.2
            recommendations.append("Verify due date calculation")

        # Business rules validation
        business_rules_issues = self._validate_business_rules(invoice)
        findings.extend(business_rules_issues)
        if business_rules_issues:
            confidence -= 0.2
            recommendations.append("Check business date rules compliance")

        # Payment terms validation
        payment_terms_issues = self._validate_payment_terms(invoice)
        findings.extend(payment_terms_issues)
        if payment_terms_issues:
            confidence -= 0.1
            recommendations.append("Verify payment terms consistency")

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _validate_invoice_date(self, invoice_date: str) -> List[Dict[str, Any]]:
        """Validate invoice date format and range"""
        issues = []

        try:
            parsed_date = datetime.fromisoformat(invoice_date.replace('/', '-'))
            today = datetime.now()

            # Check if date is too far in the future
            days_future = (parsed_date - today).days
            if days_future > self.max_future_days:
                issues.append({
                    'type': 'future_invoice_date',
                    'severity': 'high',
                    'description': f"Invoice date is {days_future} days in the future",
                    'invoice_date': invoice_date,
                    'days_future': days_future
                })

            # Check if date is too far in the past
            days_past = (today - parsed_date).days
            if days_past > self.max_past_days:
                issues.append({
                    'type': 'old_invoice_date',
                    'severity': 'medium',
                    'description': f"Invoice date is {days_past} days old",
                    'invoice_date': invoice_date,
                    'days_past': days_past
                })

            # Check for weekend dates (might indicate backdating)
            if parsed_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                issues.append({
                    'type': 'weekend_invoice_date',
                    'severity': 'low',
                    'description': f"Invoice date falls on weekend: {parsed_date.strftime('%A')}",
                    'invoice_date': invoice_date,
                    'weekday': parsed_date.strftime('%A')
                })

        except ValueError:
            issues.append({
                'type': 'invalid_invoice_date_format',
                'severity': 'high',
                'description': f"Invalid invoice date format: {invoice_date}",
                'invoice_date': invoice_date
            })

        return issues

    def _validate_due_date(self, due_date: str, invoice_date: str) -> List[Dict[str, Any]]:
        """Validate due date against invoice date"""
        issues = []

        try:
            parsed_due = datetime.fromisoformat(due_date.replace('/', '-'))
            parsed_invoice = datetime.fromisoformat(invoice_date.replace('/', '-'))

            # Due date should be after invoice date
            if parsed_due <= parsed_invoice:
                issues.append({
                    'type': 'due_date_before_invoice_date',
                    'severity': 'high',
                    'description': f"Due date {due_date} is not after invoice date {invoice_date}",
                    'due_date': due_date,
                    'invoice_date': invoice_date
                })

            # Check for reasonable payment period
            payment_days = (parsed_due - parsed_invoice).days
            if payment_days > 90:
                issues.append({
                    'type': 'extended_payment_period',
                    'severity': 'medium',
                    'description': f"Payment period of {payment_days} days is unusually long",
                    'payment_days': payment_days
                })
            elif payment_days < 1:
                issues.append({
                    'type': 'immediate_payment_required',
                    'severity': 'medium',
                    'description': f"Payment required within {payment_days} days",
                    'payment_days': payment_days
                })

        except ValueError:
            issues.append({
                'type': 'invalid_due_date_format',
                'severity': 'high',
                'description': f"Invalid due date format: {due_date}",
                'due_date': due_date
            })

        return issues

    def _validate_business_rules(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate business-specific date rules"""
        issues = []

        try:
            invoice_date = datetime.fromisoformat(invoice.invoice_date.replace('/', '-'))

            # Check for month-end processing rules
            if invoice_date.day > 25 and invoice.total_amount > 5000:
                issues.append({
                    'type': 'month_end_high_value',
                    'severity': 'medium',
                    'description': f"High-value invoice near month-end may need special approval",
                    'amount': invoice.total_amount,
                    'date': invoice.invoice_date
                })

            # Check for fiscal year end processing
            # Assuming fiscal year ends December 31
            if invoice_date.month == 12 and invoice_date.day > 20:
                issues.append({
                    'type': 'fiscal_year_end_processing',
                    'severity': 'low',
                    'description': "Invoice processed near fiscal year end",
                    'date': invoice.invoice_date
                })

        except ValueError:
            pass  # Already handled in date validation

        return issues

    def _validate_payment_terms(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate payment terms consistency with due date"""
        issues = []

        if not invoice.payment_terms:
            return issues

        try:
            invoice_date = datetime.fromisoformat(invoice.invoice_date.replace('/', '-'))
            due_date = datetime.fromisoformat(invoice.due_date.replace('/', '-'))
            actual_days = (due_date - invoice_date).days

            # Extract days from payment terms
            terms_lower = invoice.payment_terms.lower()

            # Common payment terms patterns
            if 'net 30' in terms_lower:
                expected_days = 30
            elif 'net 15' in terms_lower:
                expected_days = 15
            elif 'net 10' in terms_lower:
                expected_days = 10
            elif 'due on receipt' in terms_lower or 'immediate' in terms_lower:
                expected_days = 0
            else:
                # Try to extract number from terms
                import re
                match = re.search(r'(\d+)', terms_lower)
                expected_days = int(match.group(1)) if match else None

            if expected_days is not None and abs(actual_days - expected_days) > 2:
                issues.append({
                    'type': 'payment_terms_mismatch',
                    'severity': 'medium',
                    'description': f"Payment terms '{invoice.payment_terms}' don't match due date calculation",
                    'expected_days': expected_days,
                    'actual_days': actual_days,
                    'payment_terms': invoice.payment_terms
                })

        except (ValueError, AttributeError):
            pass

        return issues

class TaxCalculationAgent(SpecializedAgent):
    """Specialized agent for tax calculation validation and compliance"""

    def __init__(self):
        super().__init__(
            name="Tax Calculation Agent",
            description="Validates tax calculations and ensures tax compliance"
        )

        # Standard tax rates by state/jurisdiction
        self.standard_tax_rates = {
            'CA': 0.0725,  # California
            'NY': 0.08,    # New York
            'TX': 0.0625,  # Texas
            'FL': 0.06,    # Florida
            'WA': 0.065,   # Washington
            'OR': 0.0     # Oregon (no sales tax)
        }

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Validate tax calculations"""
        start_time = datetime.now()
        self.logger.info(f"Processing tax calculation for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # Basic tax calculation validation
        calculation_issues = self._validate_tax_calculation(invoice)
        findings.extend(calculation_issues)
        if calculation_issues:
            confidence -= 0.3
            recommendations.append("Verify tax calculation accuracy")

        # Tax rate validation
        rate_issues = self._validate_tax_rate(invoice)
        findings.extend(rate_issues)
        if rate_issues:
            confidence -= 0.2
            recommendations.append("Review tax rate applied")

        # Tax exemption validation
        exemption_issues = self._check_tax_exemptions(invoice)
        findings.extend(exemption_issues)
        if exemption_issues:
            confidence -= 0.1
            recommendations.append("Verify tax exemption status")

        # Jurisdiction compliance
        jurisdiction_issues = self._validate_jurisdiction_compliance(invoice)
        findings.extend(jurisdiction_issues)
        if jurisdiction_issues:
            confidence -= 0.2
            recommendations.append("Check jurisdiction-specific tax requirements")

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _validate_tax_calculation(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate basic tax calculation"""
        issues = []

        if invoice.subtotal <= 0:
            return issues

        # Calculate expected tax amount
        expected_tax = round(invoice.subtotal * 0.0725, 2)  # Default CA rate
        difference = abs(invoice.tax_amount - expected_tax)

        # Allow for different tax rates but flag large discrepancies
        if difference > (invoice.subtotal * 0.05):  # More than 5% of subtotal
            issues.append({
                'type': 'tax_calculation_discrepancy',
                'severity': 'medium',
                'description': f"Tax amount {invoice.tax_amount} differs significantly from expected range",
                'calculated_tax': expected_tax,
                'actual_tax': invoice.tax_amount,
                'difference': difference
            })

        # Check for negative tax
        if invoice.tax_amount < 0:
            issues.append({
                'type': 'negative_tax_amount',
                'severity': 'high',
                'description': f"Negative tax amount: {invoice.tax_amount}",
                'tax_amount': invoice.tax_amount
            })

        return issues

    def _validate_tax_rate(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate applied tax rate"""
        issues = []

        if invoice.subtotal <= 0:
            return issues

        actual_tax_rate = invoice.tax_amount / invoice.subtotal

        # Extract state from vendor address (simplified)
        vendor_address = invoice.vendor.get('address', '').upper()
        state_code = None
        for state in self.standard_tax_rates.keys():
            if state in vendor_address:
                state_code = state
                break

        if state_code:
            expected_rate = self.standard_tax_rates[state_code]
            rate_difference = abs(actual_tax_rate - expected_rate)

            if rate_difference > 0.02:  # More than 2% difference
                issues.append({
                    'type': 'tax_rate_mismatch',
                    'severity': 'medium',
                    'description': f"Tax rate {actual_tax_rate*100:.2f}% differs from {state_code} standard rate {expected_rate*100:.2f}%",
                    'actual_rate': actual_tax_rate,
                    'expected_rate': expected_rate,
                    'difference': rate_difference,
                    'state': state_code
                })

        # Check for exact percentages that might indicate rounding errors
        rate_percent = actual_tax_rate * 100
        if rate_percent == int(rate_percent) and rate_percent > 0:
            issues.append({
                'type': 'exact_percentage_tax_rate',
                'severity': 'low',
                'description': f"Tax rate is exact percentage: {rate_percent:.0f}% (may indicate estimation)",
                'tax_rate': actual_tax_rate,
                'exact_percentage': rate_percent
            })

        return issues

    def _check_tax_exemptions(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Check for potential tax exemptions"""
        issues = []

        # Check for zero tax with explanations
        if invoice.tax_amount == 0 and invoice.subtotal > 0:
            # Look for tax exemption indicators
            text_to_check = (invoice.notes + ' ' + ' '.join(
                item.get('description', '') for item in invoice.line_items
            )).lower()

            exemption_indicators = [
                'tax exempt', 'tax-exempt', 'exempt', 'non-taxable',
                'resale', 'wholesale', 'government', 'nonprofit', 'charity'
            ]

            has_exemption_indicator = any(indicator in text_to_check for indicator in exemption_indicators)

            if not has_exemption_indicator:
                issues.append({
                    'type': 'zero_tax_no_exemption_noted',
                    'severity': 'medium',
                    'description': f"Zero tax amount on ${invoice.subtotal:.2f} subtotal with no exemption explanation",
                    'subtotal': invoice.subtotal,
                    'recommendation': "Verify tax exemption status or add exemption documentation"
                })
            else:
                issues.append({
                    'type': 'tax_exemption_claimed',
                    'severity': 'low',
                    'description': f"Tax exemption claimed - verify documentation",
                    'exemption_indicators': [ind for ind in exemption_indicators if ind in text_to_check]
                })

        return issues

    def _validate_jurisdiction_compliance(self, invoice: InvoiceData) -> List[Dict[str, Any]]:
        """Validate jurisdiction-specific compliance"""
        issues = []

        # This would integrate with actual tax compliance systems
        # For now, implementing basic checks

        vendor_address = invoice.vendor.get('address', '').upper()

        # Check for interstate commerce (simplified)
        if 'NY' in vendor_address and invoice.total_amount > 10000:
            issues.append({
                'type': 'interstate_commerce_check',
                'severity': 'low',
                'description': "High-value interstate transaction may require additional tax review",
                'amount': invoice.total_amount,
                'vendor_location': vendor_address
            })

        return issues

class POMatchingAgent(SpecializedAgent):
    """Specialized agent for matching invoices to purchase orders"""

    def __init__(self):
        super().__init__(
            name="PO Matching Agent",
            description="Matches invoices to purchase orders and validates procurement compliance"
        )

        # Mock PO database
        self.mock_po_database = {
            'PO-2024-001': {
                'po_number': 'PO-2024-001',
                'vendor': 'ABC Office Supplies',
                'total_amount': 1300.00,
                'status': 'approved',
                'items': [
                    {'description': 'Premium Copy Paper', 'quantity': 5, 'unit_price': 45.0},
                    {'description': 'Ballpoint Pens', 'quantity': 10, 'unit_price': 12.5},
                    {'description': 'Desk Organizers', 'quantity': 20, 'unit_price': 42.5}
                ]
            },
            'PO-2024-002': {
                'po_number': 'PO-2024-002',
                'vendor': 'TechCorp Solutions',
                'total_amount': 10000.00,
                'status': 'approved',
                'items': [
                    {'description': 'Enterprise Software License', 'quantity': 1, 'unit_price': 6000.0},
                    {'description': 'Setup and Configuration', 'quantity': 25, 'unit_price': 100.0}
                ]
            }
        }

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Match invoice to purchase order"""
        start_time = datetime.now()
        self.logger.info(f"Processing PO matching for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        po_number = invoice.purchase_order

        if not po_number:
            findings.append({
                'type': 'no_po_reference',
                'severity': 'medium',
                'description': "No purchase order reference found",
                'recommendation': "Verify if PO is required for this transaction"
            })
            confidence -= 0.2
            recommendations.append("Obtain purchase order reference")
        else:
            # Match to PO database
            po_match_results = self._match_to_po_database(invoice, po_number)
            findings.extend(po_match_results['findings'])
            confidence -= po_match_results['confidence_reduction']
            recommendations.extend(po_match_results['recommendations'])

            # Validate amounts against PO
            if po_match_results.get('po_found'):
                amount_validation = self._validate_amounts_against_po(invoice, po_number)
                findings.extend(amount_validation['findings'])
                confidence -= amount_validation['confidence_reduction']
                recommendations.extend(amount_validation['recommendations'])

                # Validate line items against PO
                item_validation = self._validate_items_against_po(invoice, po_number)
                findings.extend(item_validation['findings'])
                confidence -= item_validation['confidence_reduction']
                recommendations.extend(item_validation['recommendations'])

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _match_to_po_database(self, invoice: InvoiceData, po_number: str) -> Dict[str, Any]:
        """Match invoice to PO database"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0
        po_found = False

        if po_number not in self.mock_po_database:
            findings.append({
                'type': 'po_not_found',
                'severity': 'high',
                'description': f"Purchase order {po_number} not found in system",
                'po_number': po_number
            })
            confidence_reduction = 0.4
            recommendations.append("Verify purchase order number and status")
        else:
            po_found = True
            po_data = self.mock_po_database[po_number]

            # Validate vendor match
            po_vendor = po_data['vendor']
            invoice_vendor = invoice.vendor.get('name', '')

            if po_vendor != invoice_vendor:
                findings.append({
                    'type': 'vendor_mismatch',
                    'severity': 'critical',
                    'description': f"Vendor mismatch: PO vendor '{po_vendor}' vs invoice vendor '{invoice_vendor}'",
                    'po_vendor': po_vendor,
                    'invoice_vendor': invoice_vendor
                })
                confidence_reduction = 0.5
                recommendations.append("Verify vendor information matches purchase order")

            # Check PO status
            if po_data['status'] != 'approved':
                findings.append({
                    'type': 'po_not_approved',
                    'severity': 'high',
                    'description': f"Purchase order status: {po_data['status']}",
                    'po_status': po_data['status']
                })
                confidence_reduction += 0.3
                recommendations.append("Ensure purchase order is approved before processing")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction,
            'po_found': po_found
        }

    def _validate_amounts_against_po(self, invoice: InvoiceData, po_number: str) -> Dict[str, Any]:
        """Validate invoice amounts against PO"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        po_data = self.mock_po_database.get(po_number, {})
        po_total = po_data.get('total_amount', 0)

        amount_difference = abs(invoice.total_amount - po_total)
        percentage_difference = (amount_difference / po_total) * 100 if po_total > 0 else 0

        if percentage_difference > 10:  # More than 10% difference
            findings.append({
                'type': 'amount_exceeds_po',
                'severity': 'high',
                'description': f"Invoice amount ${invoice.total_amount:.2f} differs significantly from PO amount ${po_total:.2f}",
                'invoice_amount': invoice.total_amount,
                'po_amount': po_total,
                'difference': amount_difference,
                'percentage_difference': percentage_difference
            })
            confidence_reduction = 0.3
            recommendations.append("Verify amount differences with purchase order")
        elif percentage_difference > 5:  # 5-10% difference
            findings.append({
                'type': 'amount_variance',
                'severity': 'medium',
                'description': f"Invoice amount varies from PO by {percentage_difference:.1f}%",
                'percentage_difference': percentage_difference
            })
            confidence_reduction = 0.1
            recommendations.append("Review amount variance with procurement")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

    def _validate_items_against_po(self, invoice: InvoiceData, po_number: str) -> Dict[str, Any]:
        """Validate line items against PO"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        po_data = self.mock_po_database.get(po_number, {})
        po_items = po_data.get('items', [])

        # Create simplified matching based on descriptions
        po_descriptions = [item['description'].lower() for item in po_items]

        unmatched_items = []
        for i, invoice_item in enumerate(invoice.line_items):
            item_desc = invoice_item.get('description', '').lower()

            # Simple fuzzy matching
            matched = any(
                any(word in po_desc for word in item_desc.split() if len(word) > 3)
                for po_desc in po_descriptions
            )

            if not matched:
                unmatched_items.append({
                    'line_item': i + 1,
                    'description': invoice_item.get('description', ''),
                    'amount': invoice_item.get('total', 0)
                })

        if unmatched_items:
            findings.append({
                'type': 'unmatched_line_items',
                'severity': 'medium',
                'description': f"{len(unmatched_items)} line items don't match PO items",
                'unmatched_items': unmatched_items
            })
            confidence_reduction = 0.2
            recommendations.append("Verify line items match purchase order specifications")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

class FraudDetectionAgent(SpecializedAgent):
    """Specialized agent for detecting potential fraud patterns"""

    def __init__(self):
        super().__init__(
            name="Fraud Detection Agent",
            description="Detects potential fraud patterns and anomalies in invoices"
        )
        self.fraud_patterns = self._initialize_fraud_patterns()

    def _initialize_fraud_patterns(self) -> Dict[str, Any]:
        """Initialize fraud detection patterns"""
        return {
            'round_amounts': {'threshold': 100, 'weight': 0.3},
            'duplicate_vendor_amounts': {'threshold': 0.95, 'weight': 0.7},
            'unusual_timing': {'weekend_weight': 0.2, 'late_night_weight': 0.3},
            'vendor_inconsistencies': {'weight': 0.8},
            'amount_just_under_limit': {'common_limits': [1000, 5000, 10000], 'threshold': 50, 'weight': 0.5}
        }

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Detect potential fraud patterns"""
        start_time = datetime.now()
        self.logger.info(f"Processing fraud detection for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0
        fraud_score = 0.0

        # Pattern-based fraud detection
        pattern_results = await self._detect_fraud_patterns(invoice)
        findings.extend(pattern_results['findings'])
        fraud_score += pattern_results['fraud_score']

        # Statistical anomaly detection
        anomaly_results = await self._detect_statistical_anomalies(invoice)
        findings.extend(anomaly_results['findings'])
        fraud_score += anomaly_results['fraud_score']

        # Behavioral analysis
        behavioral_results = await self._analyze_behavioral_patterns(invoice)
        findings.extend(behavioral_results['findings'])
        fraud_score += behavioral_results['fraud_score']

        # Digital forensics
        digital_results = self._analyze_digital_forensics(invoice)
        findings.extend(digital_results['findings'])
        fraud_score += digital_results['fraud_score']

        # Calculate overall fraud risk
        fraud_risk = min(1.0, fraud_score)

        # Add fraud risk assessment
        findings.append({
            'type': 'fraud_risk_assessment',
            'severity': 'critical' if fraud_risk > 0.7 else 'high' if fraud_risk > 0.4 else 'low',
            'description': f"Fraud risk score: {fraud_risk:.2f}",
            'fraud_score': fraud_risk,
            'risk_level': 'HIGH' if fraud_risk > 0.7 else 'MEDIUM' if fraud_risk > 0.4 else 'LOW'
        })

        # Adjust confidence based on fraud score
        confidence = max(0.1, 1.0 - fraud_risk)

        # Generate recommendations
        if fraud_risk > 0.7:
            recommendations.extend([
                "URGENT: Manual review required - high fraud risk detected",
                "Verify vendor authenticity and supporting documentation",
                "Consider fraud investigation procedures"
            ])
        elif fraud_risk > 0.4:
            recommendations.extend([
                "Enhanced verification recommended",
                "Review supporting documentation",
                "Consider additional approval levels"
            ])
        else:
            recommendations.append("Standard processing - low fraud risk")

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'FAIL' if fraud_risk > 0.7 else 'WARN' if fraud_risk > 0.4 else 'PASS'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    async def _detect_fraud_patterns(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Detect known fraud patterns"""
        findings = []
        fraud_score = 0.0

        # Round amount detection
        if invoice.total_amount == round(invoice.total_amount) and invoice.total_amount >= 100:
            fraud_score += self.fraud_patterns['round_amounts']['weight'] * 0.3
            findings.append({
                'type': 'round_amount_pattern',
                'severity': 'medium',
                'description': f"Round amount invoice: ${invoice.total_amount:.0f}",
                'amount': invoice.total_amount,
                'risk_factor': 'Round amounts may indicate estimated or fabricated invoices'
            })

        # Just-under-limit detection
        for limit in self.fraud_patterns['amount_just_under_limit']['common_limits']:
            threshold = self.fraud_patterns['amount_just_under_limit']['threshold']
            if limit - threshold <= invoice.total_amount < limit:
                fraud_score += self.fraud_patterns['amount_just_under_limit']['weight'] * 0.4
                findings.append({
                    'type': 'amount_just_under_limit',
                    'severity': 'high',
                    'description': f"Amount ${invoice.total_amount:.2f} just under ${limit} limit",
                    'amount': invoice.total_amount,
                    'limit': limit,
                    'risk_factor': 'Amounts just under limits may indicate deliberate evasion'
                })

        # Vendor name inconsistencies
        vendor_name = invoice.vendor.get('name', '')
        if self._check_vendor_inconsistencies(vendor_name):
            fraud_score += self.fraud_patterns['vendor_inconsistencies']['weight'] * 0.5
            findings.append({
                'type': 'vendor_name_inconsistency',
                'severity': 'high',
                'description': f"Potential vendor name manipulation detected",
                'vendor_name': vendor_name,
                'risk_factor': 'Similar vendor names may indicate impersonation'
            })

        return {'findings': findings, 'fraud_score': fraud_score}

    async def _detect_statistical_anomalies(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Detect statistical anomalies"""
        findings = []
        fraud_score = 0.0

        # Get vendor history for comparison
        vendor_history = db_manager.get_vendor_history(invoice.vendor.get('name', ''))

        if vendor_history['recent_invoices']:
            recent_amounts = [inv.get('total_amount', 0) for inv in vendor_history['recent_invoices']]

            if recent_amounts:
                avg_amount = sum(recent_amounts) / len(recent_amounts)
                std_dev = (sum((x - avg_amount) ** 2 for x in recent_amounts) / len(recent_amounts)) ** 0.5

                # Z-score calculation
                if std_dev > 0:
                    z_score = abs(invoice.total_amount - avg_amount) / std_dev

                    if z_score > 3:  # More than 3 standard deviations
                        fraud_score += 0.4
                        findings.append({
                            'type': 'statistical_outlier',
                            'severity': 'high',
                            'description': f"Invoice amount is statistical outlier (z-score: {z_score:.2f})",
                            'z_score': z_score,
                            'amount': invoice.total_amount,
                            'vendor_avg': avg_amount,
                            'risk_factor': 'Highly unusual amounts may indicate fraud'
                        })
                    elif z_score > 2:
                        fraud_score += 0.2
                        findings.append({
                            'type': 'amount_anomaly',
                            'severity': 'medium',
                            'description': f"Unusual amount for this vendor (z-score: {z_score:.2f})",
                            'z_score': z_score
                        })

        return {'findings': findings, 'fraud_score': fraud_score}

    async def _analyze_behavioral_patterns(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Analyze behavioral patterns"""
        findings = []
        fraud_score = 0.0

        # Timing analysis
        try:
            invoice_date = datetime.fromisoformat(invoice.invoice_date.replace('/', '-'))

            # Weekend submission
            if invoice_date.weekday() >= 5:
                fraud_score += self.fraud_patterns['unusual_timing']['weekend_weight']
                findings.append({
                    'type': 'weekend_submission',
                    'severity': 'low',
                    'description': f"Invoice dated on {invoice_date.strftime('%A')}",
                    'date': invoice.invoice_date,
                    'risk_factor': 'Weekend activity may indicate urgency to avoid scrutiny'
                })

            # Rush processing indicators
            notes_lower = invoice.notes.lower()
            rush_indicators = ['urgent', 'rush', 'asap', 'emergency', 'immediate']
            if any(indicator in notes_lower for indicator in rush_indicators):
                fraud_score += 0.3
                findings.append({
                    'type': 'rush_processing_request',
                    'severity': 'medium',
                    'description': "Rush processing requested",
                    'risk_factor': 'Rush requests may be used to bypass normal controls'
                })

        except ValueError:
            pass

        return {'findings': findings, 'fraud_score': fraud_score}

    def _analyze_digital_forensics(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Analyze digital forensics indicators"""
        findings = []
        fraud_score = 0.0

        # Check for potential data manipulation indicators
        # This would integrate with actual document analysis tools

        # Line item precision analysis
        line_item_precisions = []
        for item in invoice.line_items:
            try:
                total = float(item.get('total', 0))
                if total > 0:
                    decimal_places = len(str(total).split('.')[-1]) if '.' in str(total) else 0
                    line_item_precisions.append(decimal_places)
            except:
                pass

        if line_item_precisions:
            # Unusual precision patterns
            if all(p == 0 for p in line_item_precisions) and len(line_item_precisions) > 1:
                fraud_score += 0.2
                findings.append({
                    'type': 'unusual_precision_pattern',
                    'severity': 'low',
                    'description': "All line items are round numbers",
                    'risk_factor': 'Overly neat calculations may indicate manual fabrication'
                })

        # Check for suspicious vendor information patterns
        vendor_email = invoice.vendor.get('email', '')
        if vendor_email:
            # Free email domain check
            free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            if any(domain in vendor_email.lower() for domain in free_domains):
                fraud_score += 0.1
                findings.append({
                    'type': 'free_email_domain',
                    'severity': 'low',
                    'description': f"Vendor using free email service: {vendor_email}",
                    'risk_factor': 'Professional vendors typically use business email domains'
                })

        return {'findings': findings, 'fraud_score': fraud_score}

    def _check_vendor_inconsistencies(self, vendor_name: str) -> bool:
        """Check for vendor name inconsistencies"""
        # This would integrate with a comprehensive vendor database
        # For now, implement basic checks

        # Check for common manipulation patterns
        suspicious_patterns = [
            r'[0O]',  # Zero/O substitution
            r'[1lI]',  # One/l/I substitution
            r'\s{2,}',  # Multiple spaces
            r'[^\w\s&\-\.]'  # Unusual characters
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, vendor_name):
                return True

        return False

class ComplianceCheckAgent(SpecializedAgent):
    """Specialized agent for compliance and regulatory validation"""

    def __init__(self):
        super().__init__(
            name="Compliance Check Agent",
            description="Validates regulatory compliance and business rules"
        )
        self.compliance_rules = self._initialize_compliance_rules()

    def _initialize_compliance_rules(self) -> Dict[str, Any]:
        """Initialize compliance rules"""
        return {
            'sox_compliance': {
                'high_value_threshold': 10000,
                'approval_required': True,
                'documentation_required': ['contract', 'po', 'receipt']
            },
            'tax_compliance': {
                'tax_rate_validation': True,
                'exemption_documentation': True
            },
            'procurement_compliance': {
                'po_required_threshold': 1000,
                'vendor_approval_required': True,
                'competitive_bidding_threshold': 25000
            },
            'data_retention': {
                'minimum_retention_years': 7,
                'secure_storage_required': True
            }
        }

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Check compliance requirements"""
        start_time = datetime.now()
        self.logger.info(f"Processing compliance check for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # SOX Compliance
        sox_results = self._check_sox_compliance(invoice)
        findings.extend(sox_results['findings'])
        confidence -= sox_results['confidence_reduction']
        recommendations.extend(sox_results['recommendations'])

        # Tax Compliance
        tax_results = self._check_tax_compliance(invoice)
        findings.extend(tax_results['findings'])
        confidence -= tax_results['confidence_reduction']
        recommendations.extend(tax_results['recommendations'])

        # Procurement Compliance
        procurement_results = self._check_procurement_compliance(invoice)
        findings.extend(procurement_results['findings'])
        confidence -= procurement_results['confidence_reduction']
        recommendations.extend(procurement_results['recommendations'])

        # Data Privacy Compliance
        privacy_results = self._check_data_privacy_compliance(invoice)
        findings.extend(privacy_results['findings'])
        confidence -= privacy_results['confidence_reduction']
        recommendations.extend(privacy_results['recommendations'])

        # Anti-Money Laundering (AML) checks
        aml_results = self._check_aml_compliance(invoice)
        findings.extend(aml_results['findings'])
        confidence -= aml_results['confidence_reduction']
        recommendations.extend(aml_results['recommendations'])

        processing_time = (datetime.now() - start_time).total_seconds()
        status = 'PASS' if confidence > 0.7 else 'WARN' if confidence > 0.4 else 'FAIL'

        return self._create_result(status, confidence, findings, recommendations, processing_time)

    def _check_sox_compliance(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Check Sarbanes-Oxley compliance"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        sox_rules = self.compliance_rules['sox_compliance']

        # High-value transaction checks
        if invoice.total_amount >= sox_rules['high_value_threshold']:
            findings.append({
                'type': 'sox_high_value_transaction',
                'severity': 'high',
                'description': f"High-value transaction (${invoice.total_amount:.2f}) requires SOX compliance review",
                'amount': invoice.total_amount,
                'threshold': sox_rules['high_value_threshold']
            })

            # Check for required documentation
            if not invoice.purchase_order:
                findings.append({
                    'type': 'sox_missing_po',
                    'severity': 'critical',
                    'description': "Purchase order required for SOX compliance on high-value transactions",
                    'amount': invoice.total_amount
                })
                confidence_reduction += 0.3
                recommendations.append("Obtain purchase order for SOX compliance")

            # Check for proper approval documentation
            if not invoice.notes or 'approved' not in invoice.notes.lower():
                findings.append({
                    'type': 'sox_approval_documentation',
                    'severity': 'high',
                    'description': "Approval documentation may be incomplete for SOX compliance",
                    'requirement': "Documented approval process required"
                })
                confidence_reduction += 0.2
                recommendations.append("Document approval process for SOX compliance")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

    def _check_tax_compliance(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Check tax compliance requirements"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        # Tax rate reasonableness
        if invoice.subtotal > 0:
            tax_rate = invoice.tax_amount / invoice.subtotal

            # Check for unusual tax rates
            if tax_rate > 0.15:  # 15% seems high for most jurisdictions
                findings.append({
                    'type': 'tax_compliance_high_rate',
                    'severity': 'medium',
                    'description': f"Tax rate {tax_rate*100:.2f}% may require compliance review",
                    'tax_rate': tax_rate
                })
                confidence_reduction += 0.1
                recommendations.append("Verify tax rate compliance with local regulations")

        # Zero tax compliance
        if invoice.tax_amount == 0 and invoice.subtotal > 100:
            exemption_indicators = ['exempt', 'resale', 'wholesale', 'government']
            notes_text = invoice.notes.lower()

            if not any(indicator in notes_text for indicator in exemption_indicators):
                findings.append({
                    'type': 'tax_compliance_zero_tax',
                    'severity': 'medium',
                    'description': "Zero tax amount requires exemption documentation",
                    'subtotal': invoice.subtotal
                })
                confidence_reduction += 0.1
                recommendations.append("Provide tax exemption documentation")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

    def _check_procurement_compliance(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Check procurement compliance"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        procurement_rules = self.compliance_rules['procurement_compliance']

        # PO requirement check
        if (invoice.total_amount >= procurement_rules['po_required_threshold'] and
            not invoice.purchase_order):
            findings.append({
                'type': 'procurement_po_required',
                'severity': 'high',
                'description': f"Purchase order required for amounts >= ${procurement_rules['po_required_threshold']}",
                'amount': invoice.total_amount,
                'threshold': procurement_rules['po_required_threshold']
            })
            confidence_reduction += 0.3
            recommendations.append("Obtain purchase order for procurement compliance")

        # Competitive bidding threshold
        if invoice.total_amount >= procurement_rules['competitive_bidding_threshold']:
            findings.append({
                'type': 'procurement_competitive_bidding',
                'severity': 'medium',
                'description': f"Amount >= ${procurement_rules['competitive_bidding_threshold']} may require competitive bidding documentation",
                'amount': invoice.total_amount,
                'threshold': procurement_rules['competitive_bidding_threshold']
            })
            recommendations.append("Verify competitive bidding process compliance")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

    def _check_data_privacy_compliance(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Check data privacy compliance (GDPR, CCPA, etc.)"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        # Check for potential PII in invoice data
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
        ]

        # Check notes and line items for PII
        text_to_check = invoice.notes + ' ' + ' '.join(
            item.get('description', '') for item in invoice.line_items
        )

        for pattern in pii_patterns:
            if re.search(pattern, text_to_check):
                findings.append({
                    'type': 'data_privacy_pii_detected',
                    'severity': 'medium',
                    'description': "Potential PII detected in invoice data",
                    'requirement': "PII handling must comply with privacy regulations"
                })
                confidence_reduction += 0.1
                recommendations.append("Review PII handling for privacy compliance")
                break

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

    def _check_aml_compliance(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Check Anti-Money Laundering compliance"""
        findings = []
        recommendations = []
        confidence_reduction = 0.0

        # High-value transaction reporting
        if invoice.total_amount >= 10000:
            findings.append({
                'type': 'aml_high_value_transaction',
                'severity': 'medium',
                'description': f"High-value transaction (${invoice.total_amount:.2f}) may require AML reporting",
                'amount': invoice.total_amount,
                'threshold': 10000
            })
            recommendations.append("Review AML reporting requirements")

        # Multiple transactions check (would require additional context)
        vendor_name = invoice.vendor.get('name', '')
        vendor_history = db_manager.get_vendor_history(vendor_name)

        if vendor_history['recent_invoices']:
            recent_total = sum(
                inv.get('total_amount', 0)
                for inv in vendor_history['recent_invoices'][-10:]  # Last 10 transactions
            )

            if recent_total >= 25000:  # $25K in recent transactions
                findings.append({
                    'type': 'aml_cumulative_transactions',
                    'severity': 'medium',
                    'description': f"Cumulative recent transactions: ${recent_total:.2f}",
                    'cumulative_amount': recent_total,
                    'vendor': vendor_name
                })
                recommendations.append("Review cumulative transaction patterns for AML compliance")

        return {
            'findings': findings,
            'recommendations': recommendations,
            'confidence_reduction': confidence_reduction
        }

class AnalyticsEngineAgent(SpecializedAgent):
    """Specialized agent for analytics and reporting"""

    def __init__(self):
        super().__init__(
            name="Analytics Engine Agent",
            description="Generates analytics and insights from invoice verification data"
        )

    async def process(self, invoice: InvoiceData, context: Dict[str, Any] = None) -> VerificationResult:
        """Generate analytics and insights"""
        start_time = datetime.now()
        self.logger.info(f"Processing analytics for invoice {invoice.invoice_number}")

        findings = []
        recommendations = []
        confidence = 1.0

        # Generate verification summary
        verification_results = context.get('verification_results', []) if context else []
        summary = self._generate_verification_summary(verification_results)

        # Generate vendor analytics
        vendor_analytics = await self._generate_vendor_analytics(invoice)

        # Generate system performance metrics
        performance_metrics = await self._generate_performance_metrics()

        # Generate trend analysis
        trend_analysis = await self._generate_trend_analysis()

        # Compile comprehensive analytics
        analytics_data = {
            'verification_summary': summary,
            'vendor_analytics': vendor_analytics,
            'performance_metrics': performance_metrics,
            'trend_analysis': trend_analysis,
            'generated_at': datetime.now().isoformat()
        }

        findings.append({
            'type': 'analytics_report',
            'severity': 'info',
            'description': 'Comprehensive analytics report generated',
            'data': analytics_data
        })

        recommendations.extend([
            "Review verification patterns for system improvement",
            "Monitor vendor performance trends",
            "Track compliance metrics regularly"
        ])

        processing_time = (datetime.now() - start_time).total_seconds()

        return self._create_result('PASS', confidence, findings, recommendations, processing_time)

    def _generate_verification_summary(self, verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Generate summary of verification results"""
        if not verification_results:
            return {'message': 'No verification results to analyze'}

        # Count by status
        status_counts = Counter(result.status for result in verification_results)

        # Average confidence
        avg_confidence = sum(result.confidence for result in verification_results) / len(verification_results)

        # Agent performance
        agent_performance = {}
        for result in verification_results:
            agent_name = result.agent_name
            if agent_name not in agent_performance:
                agent_performance[agent_name] = {
                    'total_processed': 0,
                    'avg_confidence': 0,
                    'avg_processing_time': 0,
                    'status_distribution': {}
                }

            perf = agent_performance[agent_name]
            perf['total_processed'] += 1
            perf['avg_confidence'] = (
                (perf['avg_confidence'] * (perf['total_processed'] - 1) + result.confidence) /
                perf['total_processed']
            )
            perf['avg_processing_time'] = (
                (perf['avg_processing_time'] * (perf['total_processed'] - 1) + result.processing_time) /
                perf['total_processed']
            )

            status = result.status
            perf['status_distribution'][status] = perf['status_distribution'].get(status, 0) + 1

        return {
            'total_verifications': len(verification_results),
            'status_distribution': dict(status_counts),
            'average_confidence': round(avg_confidence, 3),
            'agent_performance': agent_performance
        }

    async def _generate_vendor_analytics(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Generate vendor-specific analytics"""
        vendor_name = invoice.vendor.get('name', '')
        vendor_history = db_manager.get_vendor_history(vendor_name)

        analytics = {
            'vendor_name': vendor_name,
            'historical_summary': vendor_history.get('summary', {}),
            'recent_invoices': vendor_history.get('recent_invoices', [])
        }

        # Calculate vendor risk score
        summary = vendor_history.get('summary', {})
        total_invoices = summary.get('total_invoices', 0)
        avg_confidence = summary.get('avg_confidence', 0.5)

        if total_invoices == 0:
            risk_score = 0.5  # Neutral for new vendors
        else:
            # Higher risk for low confidence, adjusted by volume
            risk_score = 1 - avg_confidence
            if total_invoices < 5:
                risk_score += 0.1  # Increase risk for low-volume vendors

        analytics['risk_assessment'] = {
            'risk_score': min(1.0, max(0.0, risk_score)),
            'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'LOW',
            'factors': {
                'transaction_volume': total_invoices,
                'average_confidence': avg_confidence,
                'is_new_vendor': total_invoices == 0
            }
        }

        return analytics

    async def _generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate system performance metrics"""
        try:
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Overall system metrics
                total_invoices = conn.execute("SELECT COUNT(*) as count FROM invoices").fetchone()['count']

                # Average confidence score
                avg_confidence = conn.execute("""
                    SELECT AVG(confidence_score) as avg_confidence
                    FROM invoices
                    WHERE confidence_score IS NOT NULL
                """).fetchone()['avg_confidence']

                # Processing time metrics (would need verification_results table)
                # For now, using mock data

                # Monthly processing volume
                monthly_volume = conn.execute("""
                    SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count
                    FROM invoices
                    GROUP BY strftime('%Y-%m', created_at)
                    ORDER BY month DESC
                    LIMIT 6
                """).fetchall()

                return {
                    'total_invoices_processed': total_invoices,
                    'average_confidence_score': round(avg_confidence or 0, 3),
                    'monthly_processing_volume': [dict(row) for row in monthly_volume],
                    'system_uptime': '99.9%',  # Mock value
                    'average_processing_time': '2.3 seconds'  # Mock value
                }

        except Exception as e:
            self.logger.error(f"Error generating performance metrics: {e}")
            return {'error': 'Unable to generate performance metrics'}

    async def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate trend analysis"""
        try:
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Monthly invoice volume and amount trends
                monthly_trends = conn.execute("""
                    SELECT strftime('%Y-%m', created_at) as month,
                           COUNT(*) as invoice_count,
                           SUM(total_amount) as total_amount,
                           AVG(total_amount) as avg_amount
                    FROM invoices
                    GROUP BY strftime('%Y-%m', created_at)
                    ORDER BY month DESC
                    LIMIT 12
                """).fetchall()

                # Vendor trend analysis
                vendor_trends = conn.execute("""
                    SELECT vendor_name,
                           COUNT(*) as invoice_count,
                           SUM(total_amount) as total_amount,
                           AVG(confidence_score) as avg_confidence
                    FROM invoices
                    WHERE created_at >= date('now', '-6 months')
                    GROUP BY vendor_name
                    ORDER BY invoice_count DESC
                    LIMIT 10
                """).fetchall()

                return {
                    'monthly_volume_trends': [dict(row) for row in monthly_trends],
                    'top_vendor_trends': [dict(row) for row in vendor_trends],
                    'trends_summary': {
                        'volume_trend': 'increasing',  # Mock analysis
                        'amount_trend': 'stable',
                        'quality_trend': 'improving'
                    }
                }

        except Exception as e:
            self.logger.error(f"Error generating trend analysis: {e}")
            return {'error': 'Unable to generate trend analysis'}

class InvoiceVerificationOrchestrator:
    """Main orchestrator that coordinates all specialized agents"""

    def __init__(self):
        self.logger = invoice_logger.get_logger('orchestrator')
        self.agents = self._initialize_agents()
        self.processing_stats = {
            'total_processed': 0,
            'success_rate': 0.0,
            'average_processing_time': 0.0
        }

    def _initialize_agents(self) -> Dict[str, SpecializedAgent]:
        """Initialize all specialized agents"""
        agents = {
            'data_extraction': DataExtractionAgent(),
            'duplicate_detection': DuplicateDetectionAgent(),
            'amount_validation': AmountValidationAgent(),
            'vendor_verification': VendorVerificationAgent(),
            'date_validation': DateValidationAgent(),
            'tax_calculation': TaxCalculationAgent(),
            'po_matching': POMatchingAgent(),
            'fraud_detection': FraudDetectionAgent(),
            'compliance_check': ComplianceCheckAgent(),
            'analytics_engine': AnalyticsEngineAgent()
        }

        self.logger.info(f"Initialized {len(agents)} specialized agents")
        return agents

    async def process_invoice(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process invoice through all specialized agents"""
        start_time = datetime.now()

        # Convert to InvoiceData object
        invoice = InvoiceData.from_dict(invoice_data)

        self.logger.info(f"Starting verification for invoice {invoice.invoice_number}")

        # Process through each agent
        verification_results = []
        overall_confidence = 1.0
        critical_failures = []

        for agent_name, agent in self.agents.items():
            try:
                self.logger.info(f"Processing with {agent_name}")

                context = {'verification_results': verification_results} if agent_name == 'analytics_engine' else None
                result = await agent.process(invoice, context)

                verification_results.append(result)

                # Track critical failures
                if result.status == 'FAIL':
                    critical_failures.append(agent_name)

                # Update overall confidence (take minimum for critical issues)
                if agent_name in ['fraud_detection', 'compliance_check']:
                    overall_confidence = min(overall_confidence, result.confidence)
                else:
                    overall_confidence *= result.confidence

                # Save result to database (in real implementation)
                # await self._save_verification_result(invoice.invoice_number, result)

            except Exception as e:
                self.logger.error(f"Error processing with {agent_name}: {e}")

                # Create error result
                error_result = VerificationResult(
                    agent_name=agent_name,
                    status='FAIL',
                    confidence=0.0,
                    findings=[{
                        'type': 'processing_error',
                        'severity': 'critical',
                        'description': f"Error in {agent_name}: {str(e)}"
                    }],
                    recommendations=[f"Manual review required for {agent_name}"],
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat()
                )
                verification_results.append(error_result)
                critical_failures.append(agent_name)

        # Generate final assessment
        processing_time = (datetime.now() - start_time).total_seconds()

        final_status = self._determine_final_status(verification_results, critical_failures)

        result = {
            'invoice_number': invoice.invoice_number,
            'overall_status': final_status,
            'overall_confidence': round(overall_confidence, 3),
            'critical_failures': critical_failures,
            'processing_time': processing_time,
            'verification_results': [result.to_dict() for result in verification_results],
            'summary': self._generate_summary(verification_results),
            'recommendations': self._generate_final_recommendations(verification_results)
        }

        # Update processing stats
        self.processing_stats['total_processed'] += 1

        self.logger.info(f"Completed verification for invoice {invoice.invoice_number}: {final_status}")

        return result

    def _determine_final_status(self, results: List[VerificationResult], critical_failures: List[str]) -> str:
        """Determine final processing status"""
        if critical_failures:
            return 'REJECTED'

        fail_count = sum(1 for r in results if r.status == 'FAIL')
        warn_count = sum(1 for r in results if r.status == 'WARN')

        if fail_count > 0:
            return 'REJECTED'
        elif warn_count > 2:  # More than 2 warnings
            return 'REVIEW_REQUIRED'
        elif warn_count > 0:
            return 'APPROVED_WITH_CONDITIONS'
        else:
            return 'APPROVED'

    def _generate_summary(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Generate verification summary"""
        total_findings = sum(len(r.findings) for r in results)

        # Categorize findings by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}

        for result in results:
            for finding in result.findings:
                severity = finding.get('severity', 'medium')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_agents_processed': len(results),
            'total_findings': total_findings,
            'findings_by_severity': severity_counts,
            'agents_status': {r.agent_name: r.status for r in results}
        }

    def _generate_final_recommendations(self, results: List[VerificationResult]) -> List[str]:
        """Generate final recommendations"""
        all_recommendations = []

        for result in results:
            all_recommendations.extend(result.recommendations)

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

# Testing functions
async def test_specialized_agents():
    """Test the specialized agents system"""
    print("🚀 Testing Specialized Invoice Verification Agents")
    print("=" * 60)

    # Create test invoice
    test_invoice_data = {
        'invoice_number': 'INV-2024-001',
        'vendor': {
            'name': 'ABC Office Supplies',
            'address': '123 Business St, San Francisco, CA 94102',
            'phone': '+1-555-0123',
            'email': 'billing@abcoffice.com',
            'tax_id': '12-3456789'
        },
        'invoice_date': '2024-01-15',
        'due_date': '2024-02-14',
        'subtotal': 1200.00,
        'tax_amount': 87.00,
        'total_amount': 1287.00,
        'payment_terms': 'Net 30',
        'purchase_order': 'PO-2024-001',
        'line_items': [
            {'description': 'Premium Copy Paper', 'quantity': 5, 'unit_price': 45.0, 'total': 225.0},
            {'description': 'Ballpoint Pens', 'quantity': 10, 'unit_price': 12.5, 'total': 125.0},
            {'description': 'Desk Organizers', 'quantity': 20, 'unit_price': 42.5, 'total': 850.0}
        ],
        'notes': 'Standard office supplies order'
    }

    # Initialize orchestrator
    orchestrator = InvoiceVerificationOrchestrator()

    # Process invoice
    result = await orchestrator.process_invoice(test_invoice_data)

    # Display results
    print(f"\n📊 Verification Results for {result['invoice_number']}")
    print(f"Overall Status: {result['overall_status']}")
    print(f"Overall Confidence: {result['overall_confidence']:.3f}")
    print(f"Processing Time: {result['processing_time']:.2f} seconds")

    print(f"\n📈 Summary:")
    summary = result['summary']
    print(f"  • Agents Processed: {summary['total_agents_processed']}")
    print(f"  • Total Findings: {summary['total_findings']}")
    print(f"  • Findings by Severity: {summary['findings_by_severity']}")

    print(f"\n🤖 Agent Status:")
    for agent, status in summary['agents_status'].items():
        status_icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
        print(f"  {status_icon} {agent}: {status}")

    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(result['recommendations'][:5], 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 60)
    print("✅ Specialized agents testing completed!")

if __name__ == "__main__":
    asyncio.run(test_specialized_agents())