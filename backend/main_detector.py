#!/usr/bin/env python3
"""
Main invoice fraud detector with complete error recovery system.

This is the unified entry point that brings together all components:
- Agent definitions with DSPy signatures
- Error validation and recovery
- Invoice processing with retry loops
- Comprehensive error handling

Usage:
    python main_detector.py --invoice "invoice_data_here"
    python main_detector.py --file invoice.json
    python main_detector.py --demo  # Run with demo data
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from invoice_processor import InvoiceProcessor, ProcessingResult
    from agent_definitions import FRAUD_DETECTION_AGENTS
    from error_validation import ErrorValidator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("- agent_definitions.py")
    print("- error_validation.py") 
    print("- invoice_processor.py")
    sys.exit(1)

# Configure logging with Windows-safe formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fraud_detection.log", encoding='utf-8')
    ]
)

# Set console encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
log = logging.getLogger("main_detector")

class InvoiceFraudDetector:
    """Main facade class for the invoice fraud detection system"""
    
    def __init__(self, max_retries: int = 3, backoff_delay: float = 1.0):
        self.processor = InvoiceProcessor(max_retries, backoff_delay)
        self.error_validator = ErrorValidator(max_retries)
        
    def analyze_invoice(self, invoice_data: str) -> dict:
        """
        Analyze invoice for fraud with comprehensive error recovery.
        
        Args:
            invoice_data: Raw invoice data as string
            
        Returns:
            Dictionary with analysis results and metadata
        """
        log.info("🔍 Starting comprehensive invoice fraud analysis...")
        
        try:
            # Validate input
            if not invoice_data or len(invoice_data.strip()) < 10:
                raise ValueError("Invoice data is empty or too short")
            
            # Process with error recovery
            result = self.processor.process_invoice(invoice_data)
            
            # Format final output
            return self._format_final_result(result)
            
        except Exception as e:
            log.error(f"Critical failure in main analyzer: {str(e)}")
            return self._create_emergency_fallback(str(e))
    
    def _format_final_result(self, result: ProcessingResult) -> dict:
        """Format processing result for output"""
        return {
            "success": result.success,
            "fraud_assessment": {
                "overall_risk": result.overall_risk,
                "risk_level": self._get_risk_level_description(result.overall_risk),
                "recommendation": result.recommendation,
                "confidence": self._calculate_overall_confidence(result.agent_details)
            },
            "summary": result.summary,
            "top_concerns": result.top_concerns,
            "next_steps": result.next_steps,
            "agent_analysis": {
                "agents_consulted": result.agent_details,
                "statistics": result.statistics
            },
            "processing_metadata": {
                "processing_time_seconds": round(result.processing_time, 2),
                "total_errors_encountered": result.error_count,
                "retry_attempts": result.retry_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def _get_risk_level_description(self, risk_score: int) -> str:
        """Get human-readable risk level description"""
        if risk_score <= 2:
            return "Very Low Risk"
        elif risk_score <= 4:
            return "Low Risk"
        elif risk_score <= 6:
            return "Medium Risk"
        elif risk_score <= 8:
            return "High Risk"
        else:
            return "Critical Risk"
    
    def _calculate_overall_confidence(self, agent_details: list) -> str:
        """Calculate overall confidence description"""
        if not agent_details:
            return "No Confidence Data"
        
        avg_confidence = sum(a.get("confidence", 0) for a in agent_details) / len(agent_details)
        
        if avg_confidence >= 8:
            return "High Confidence"
        elif avg_confidence >= 6:
            return "Medium Confidence"
        elif avg_confidence >= 4:
            return "Low-Medium Confidence"
        else:
            return "Low Confidence"
    
    def _create_emergency_fallback(self, error_msg: str) -> dict:
        """Create emergency fallback response when everything fails"""
        return {
            "success": False,
            "fraud_assessment": {
                "overall_risk": 8,
                "risk_level": "High Risk",
                "recommendation": "MANUAL_REVIEW",
                "confidence": "System Error - No Confidence Data"
            },
            "summary": f"Critical system failure prevented automated analysis: {error_msg}",
            "top_concerns": ["System processing failure", "Unable to complete automated analysis"],
            "next_steps": "Immediate manual review required due to system failure",
            "agent_analysis": {
                "agents_consulted": [],
                "statistics": {"error": "Complete system failure"}
            },
            "processing_metadata": {
                "processing_time_seconds": 0,
                "total_errors_encountered": 1,
                "retry_attempts": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "critical_error": error_msg
            }
        }

def get_demo_invoice() -> str:
    """Get demo invoice for testing"""
    return """
INVOICE #INV-2024-001

From: QuickSupplies LLC
123 Business Ave, Anytown, ST 12345
Phone: (555) 123-4567
Email: billing@quicksupplies.com

To: ABC Corporation  
456 Corporate Blvd, Business City, ST 67890

Invoice Date: 2024-03-15
Due Date: 2024-04-15
Payment Terms: Net 30

ITEMS:
1. Office Paper (500 sheets x 10 reams): $47.50
2. Printer Ink Cartridges (Black x 5): $89.99  
3. Pens (Blue, Box of 50): $12.99
4. Staplers (Heavy Duty x 3): $45.00
5. Miscellaneous Office Supplies: $2,651.52

Subtotal: $2,847.00
Shipping & Handling: $150.00
Tax (8.5%): $254.95
TOTAL: $3,251.95

Payment Instructions:
Wire Transfer to: First National Bank
Account: 1234567890
Routing: 987654321
"""

def load_invoice_from_file(filepath: str) -> str:
    """Load invoice data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
                return json.dumps(data, indent=2)
            else:
                return f.read()
    except Exception as e:
        raise ValueError(f"Failed to load invoice from {filepath}: {str(e)}")

def print_results(results: dict, output_file: str = None):
    """Print formatted results to console and optionally save to file"""
    
    print("\n" + "="*70)
    print("🕵️  INVOICE FRAUD DETECTION RESULTS")
    print("="*70)
    
    # Main assessment
    assessment = results["fraud_assessment"]
    print(f"📊 Overall Risk: {assessment['overall_risk']}/10 ({assessment['risk_level']})")
    print(f"🎯 Recommendation: {assessment['recommendation']}")
    print(f"🎲 Confidence: {assessment['confidence']}")
    print(f"📝 Summary: {results['summary']}")
    
    # Top concerns
    if results.get('top_concerns') and results['top_concerns'] != ['None']:
        print(f"\n⚠️  Top Concerns:")
        for i, concern in enumerate(results['top_concerns'], 1):
            print(f"   {i}. {concern}")
    else:
        print(f"\n✅ No major concerns identified")
    
    print(f"\n📋 Next Steps: {results['next_steps']}")
    
    # Agent details
    agent_analysis = results.get("agent_analysis", {})
    agents = agent_analysis.get("agents_consulted", [])
    
    if agents:
        print(f"\n🤖 Agent Analysis Details:")
        for agent in agents:
            print(f"   • {agent['agent'].replace('_', ' ').title()}:")
            print(f"     Risk: {agent['risk_score']}/10, Confidence: {agent['confidence']}/10")
            print(f"     Analysis: {agent['analysis'][:100]}{'...' if len(agent['analysis']) > 100 else ''}")
            if agent.get('red_flags'):
                print(f"     Red Flags: {', '.join(agent['red_flags'])}")
    
    # Statistics
    stats = agent_analysis.get("statistics", {})
    if stats and not stats.get("error"):
        print(f"\n📈 Statistics:")
        print(f"   Agents Consulted: {stats.get('agents_consulted', 0)}")
        print(f"   Average Risk: {stats.get('average_risk', 0)}/10")
        print(f"   Average Confidence: {stats.get('average_confidence', 0)}/10")
        print(f"   Total Red Flags: {stats.get('total_red_flags', 0)}")
    
    # Processing metadata
    metadata = results.get("processing_metadata", {})
    print(f"\n⏱️  Processing Info:")
    print(f"   Time: {metadata.get('processing_time_seconds', 0)}s")
    print(f"   Errors: {metadata.get('total_errors_encountered', 0)}")
    print(f"   Retries: {metadata.get('retry_attempts', 0)}")
    print(f"   Timestamp: {metadata.get('timestamp', 'Unknown')}")
    
    if metadata.get('critical_error'):
        print(f"   ❌ Critical Error: {metadata['critical_error']}")
    
    # Success indicator
    if results['success']:
        print(f"\n✅ Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis completed with errors - manual review recommended")
    
    print("="*70)
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Failed to save results to {output_file}: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Invoice Fraud Detection with Error Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_detector.py --demo
  python main_detector.py --invoice "INVOICE DATA HERE"
  python main_detector.py --file invoice.txt
  python main_detector.py --file invoice.json --output results.json
  python main_detector.py --demo --max-retries 5 --verbose

Available Agents:
""" + "\n".join([f"  • {name}: {config['description']}" 
                for name, config in FRAUD_DETECTION_AGENTS.items()])
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--invoice", help="Invoice data as string")
    input_group.add_argument("--file", help="File containing invoice data (text or JSON)")
    input_group.add_argument("--demo", action="store_true", help="Use demo invoice data")
    
    # Output options
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Processing options
    parser.add_argument("--max-retries", type=int, default=3, 
                       help="Maximum retry attempts per step (default: 3)")
    parser.add_argument("--backoff-delay", type=float, default=1.0,
                       help="Base delay between retries in seconds (default: 1.0)")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress most output")
    parser.add_argument("--log-file", default="fraud_detection.log",
                       help="Log file path (default: fraud_detection.log)")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Validate arguments
    if args.max_retries < 0 or args.max_retries > 10:
        print("❌ Error: max-retries must be between 0 and 10")
        return 1
    
    if args.backoff_delay < 0 or args.backoff_delay > 60:
        print("❌ Error: backoff-delay must be between 0 and 60 seconds")
        return 1
    
    # Get invoice data
    try:
        if args.demo:
            print("📋 Using demo invoice data...")
            invoice_data = get_demo_invoice()
        elif args.file:
            print(f"📁 Loading invoice from: {args.file}")
            invoice_data = load_invoice_from_file(args.file)
        else:
            invoice_data = args.invoice
        
        if not args.quiet:
            print(f"📄 Invoice data loaded ({len(invoice_data)} characters)")
            
    except Exception as e:
        print(f"❌ Error loading invoice data: {str(e)}")
        return 1
    
    # Initialize detector
    try:
        if not args.quiet:
            print(f"🚀 Initializing fraud detector (max_retries={args.max_retries}, backoff={args.backoff_delay}s)")
        
        detector = InvoiceFraudDetector(
            max_retries=args.max_retries,
            backoff_delay=args.backoff_delay
        )
        
    except Exception as e:
        print(f"❌ Error initializing detector: {str(e)}")
        print("💡 Make sure you have:")
        print("   • GOOGLE_API_KEY set in your .env file")
        print("   • google-generativeai installed: pip install google-generativeai")
        print("   • dspy-ai installed: pip install dspy-ai")
        return 1
    
    # Run analysis
    try:
        start_time = time.time()
        results = detector.analyze_invoice(invoice_data)
        total_time = time.time() - start_time
        
        if not args.quiet:
            print(f"⏱️  Total analysis time: {total_time:.2f} seconds")
        
        # Print results
        print_results(results, args.output)
        
        # Return appropriate exit code
        if results['success']:
            if results['fraud_assessment']['overall_risk'] >= 8:
                return 2  # High risk detected
            elif results['fraud_assessment']['overall_risk'] >= 6:
                return 1  # Medium-high risk detected
            else:
                return 0  # Low risk
        else:
            return 3  # System failure
            
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Critical error during analysis: {str(e)}")
        log.exception("Critical error in main analysis")
        return 1

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import google.generativeai
    except ImportError:
        missing_deps.append("google-generativeai")
    
    try:
        import dspy
    except ImportError:
        missing_deps.append("dspy-ai")
    
    try:
        import dotenv
    except ImportError:
        missing_deps.append("python-dotenv")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

def check_api_key():
    """Check if API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API key
    if os.getenv("GOOGLE_API_KEY"):
        return True
    
    # Check for numbered keys
    for i in range(10):
        if os.getenv(f"GOOGLE_API_KEY_{i}"):
            return True
    
    print("❌ No Google API key found!")
    print("💡 Add GOOGLE_API_KEY to your .env file:")
    print("   GOOGLE_API_KEY=your_api_key_here")
    print("\nOr use numbered keys:")
    print("   GOOGLE_API_KEY_0=your_first_key")
    print("   GOOGLE_API_KEY_1=your_second_key")
    return False

if __name__ == "__main__":
    # Pre-flight checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_api_key():
        sys.exit(1)
    
    # Run main program
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logging.exception("Unexpected error in main")
        sys.exit(1)