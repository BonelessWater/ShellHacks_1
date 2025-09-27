#!/usr/bin/env python3
"""
Multi-Agent Invoice Fraud Detection System using Google Generative AI

Architecture:
1. Core LLM analyzes invoice and determines which specialist agents to summon
2. Specialist agents analyze specific aspects of the invoice
3. Compiler LLM synthesizes all agent outputs into final fraud assessment

Usage:
    python invoice_fraud_detector.py --invoice "invoice_data_here"
    python invoice_fraud_detector.py --file invoice.json
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("invoice_fraud_detector")

@dataclass
class AgentResponse:
    agent_type: str
    analysis: str
    risk_score: int  # 1-10 scale
    confidence: int  # 1-10 scale
    red_flags: List[str]

class InvoiceFraudDetector:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.genai = self._setup_genai()
        
        # Available specialist agents
        self.available_agents = {
            "amount_validator": "Analyzes invoice amounts for unusual patterns, duplicates, or suspicious values",
            "vendor_authenticator": "Validates vendor information, checks for shell companies or suspicious entities",
            "date_analyzer": "Examines dates for logical consistency and timeline issues",
            "tax_calculator": "Verifies tax calculations and rates for accuracy",
            "format_inspector": "Checks invoice format, structure, and professional appearance",
            "payment_terms_checker": "Analyzes payment terms and conditions for red flags",
            "line_item_validator": "Reviews individual line items for reasonableness and pricing"
        }

    def _get_api_key(self) -> str:
        """Get Google API key from environment"""
        load_dotenv()
        
        # Try main key first
        key = os.getenv("GOOGLE_API_KEY")
        if key and key.strip():
            return key.strip()
            
        # Try numbered keys
        for i in range(10):
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key and key.strip():
                return key.strip()
                
        raise ValueError("No valid Google API key found. Set GOOGLE_API_KEY in .env file")

    def _setup_genai(self):
        """Initialize Google Generative AI"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def _get_model(self):
        """Get preferred Gemini model"""
        try:
            return self.genai.GenerativeModel("models/gemini-2.0-flash-exp")
        except:
            try:
                return self.genai.GenerativeModel("models/gemini-1.5-flash")
            except:
                # Fallback to any available model
                models = list(self.genai.list_models())
                if models:
                    return self.genai.GenerativeModel(models[0].name)
                raise RuntimeError("No available Gemini models found")

    def analyze_invoice(self, invoice_data: str) -> Dict[str, Any]:
        """Main method to analyze invoice for fraud"""
        log.info("🔍 Starting invoice fraud analysis...")
        
        # Step 1: Core LLM determines which agents to summon
        agents_to_summon = self._determine_agents(invoice_data)
        log.info(f"📋 Summoning {len(agents_to_summon)} specialist agents: {', '.join(agents_to_summon)}")
        
        # Step 2: Run specialist agents
        agent_responses = []
        for agent_type in agents_to_summon:
            response = self._run_specialist_agent(agent_type, invoice_data)
            agent_responses.append(response)
            log.info(f"✅ {agent_type}: Risk {response.risk_score}/10, Confidence {response.confidence}/10")
        
        # Step 3: Compiler LLM synthesizes results
        final_assessment = self._compile_results(invoice_data, agent_responses)
        
        return final_assessment

    def _determine_agents(self, invoice_data: str) -> List[str]:
        """Core LLM determines which specialist agents to summon"""
        model = self._get_model()
        
        agents_list = "\n".join([f"- {k}: {v}" for k, v in self.available_agents.items()])
        
        prompt = f"""
You are a fraud detection coordinator. Analyze this invoice data and determine which specialist agents should examine it.

INVOICE DATA:
{invoice_data}

AVAILABLE SPECIALIST AGENTS:
{agents_list}

Based on the invoice content, select 3-5 most relevant agents that should analyze this invoice for potential fraud.
Respond with ONLY a JSON list of agent names, like: ["agent1", "agent2", "agent3"]

Consider:
- What aspects of this invoice seem most important to verify?
- What red flags might exist that need specialist attention?
- Which agents would provide the most comprehensive fraud detection coverage?
"""

        try:
            response = model.generate_content(prompt)
            agents_text = response.text.strip()
            
            # Extract JSON from response
            if "[" in agents_text and "]" in agents_text:
                start = agents_text.find("[")
                end = agents_text.rfind("]") + 1
                agents_json = agents_text[start:end]
                selected_agents = json.loads(agents_json)
                
                # Validate agents exist
                valid_agents = [a for a in selected_agents if a in self.available_agents]
                return valid_agents if valid_agents else ["amount_validator", "vendor_authenticator", "format_inspector"]
            else:
                # Fallback if JSON parsing fails
                return ["amount_validator", "vendor_authenticator", "format_inspector"]
                
        except Exception as e:
            log.warning(f"Core agent selection failed: {e}. Using default agents.")
            return ["amount_validator", "vendor_authenticator", "format_inspector"]

    def _run_specialist_agent(self, agent_type: str, invoice_data: str) -> AgentResponse:
        """Run a specific specialist agent"""
        model = self._get_model()
        agent_description = self.available_agents[agent_type]
        
        prompt = f"""
You are a specialist fraud detection agent: {agent_type}
Your expertise: {agent_description}

INVOICE DATA TO ANALYZE:
{invoice_data}

Analyze this invoice data focusing ONLY on your area of expertise. Provide:

1. ANALYSIS: Detailed analysis of your specific domain (2-3 sentences)
2. RISK_SCORE: Rate fraud risk from 1-10 (1=very low risk, 10=very high risk)
3. CONFIDENCE: Rate your confidence in this assessment from 1-10 (1=low confidence, 10=high confidence)
4. RED_FLAGS: List any specific red flags you found (or "None" if no red flags)

Format your response as:
ANALYSIS: [your analysis]
RISK_SCORE: [1-10]
CONFIDENCE: [1-10]
RED_FLAGS: [flag1, flag2, flag3 or None]
"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Parse response
            analysis = self._extract_field(text, "ANALYSIS")
            risk_score = int(self._extract_field(text, "RISK_SCORE", "5"))
            confidence = int(self._extract_field(text, "CONFIDENCE", "5"))
            red_flags_text = self._extract_field(text, "RED_FLAGS", "None")
            
            red_flags = []
            if red_flags_text.lower() != "none":
                red_flags = [flag.strip() for flag in red_flags_text.split(",")]
            
            return AgentResponse(
                agent_type=agent_type,
                analysis=analysis,
                risk_score=min(10, max(1, risk_score)),
                confidence=min(10, max(1, confidence)),
                red_flags=red_flags
            )
            
        except Exception as e:
            log.error(f"Agent {agent_type} failed: {e}")
            return AgentResponse(
                agent_type=agent_type,
                analysis=f"Analysis failed: {str(e)}",
                risk_score=5,
                confidence=1,
                red_flags=["Analysis error"]
            )

    def _extract_field(self, text: str, field: str, default: str = "Unknown") -> str:
        """Extract a field from agent response"""
        try:
            lines = text.split('\n')
            for line in lines:
                if line.strip().startswith(f"{field}:"):
                    return line.split(":", 1)[1].strip()
            return default
        except:
            return default

    def _compile_results(self, invoice_data: str, agent_responses: List[AgentResponse]) -> Dict[str, Any]:
        """Compiler LLM synthesizes all agent results"""
        model = self._get_model()
        
        # Prepare agent results summary
        agent_summary = ""
        total_risk = 0
        total_confidence = 0
        all_red_flags = []
        
        for response in agent_responses:
            agent_summary += f"\n{response.agent_type.upper()}:\n"
            agent_summary += f"  Analysis: {response.analysis}\n"
            agent_summary += f"  Risk Score: {response.risk_score}/10\n"
            agent_summary += f"  Confidence: {response.confidence}/10\n"
            agent_summary += f"  Red Flags: {', '.join(response.red_flags) if response.red_flags else 'None'}\n"
            
            total_risk += response.risk_score
            total_confidence += response.confidence
            all_red_flags.extend(response.red_flags)
        
        avg_risk = total_risk / len(agent_responses) if agent_responses else 5
        avg_confidence = total_confidence / len(agent_responses) if agent_responses else 5
        
        prompt = f"""
You are the final fraud assessment compiler. Synthesize the specialist agent analyses into a comprehensive fraud assessment.

ORIGINAL INVOICE DATA:
{invoice_data}

SPECIALIST AGENT ANALYSES:
{agent_summary}

STATISTICAL SUMMARY:
- Average Risk Score: {avg_risk:.1f}/10
- Average Confidence: {avg_confidence:.1f}/10
- Total Red Flags: {len(all_red_flags)}

Provide a final assessment with:

1. OVERALL_RISK: Final fraud risk score (1-10)
2. RECOMMENDATION: APPROVE, REVIEW, or REJECT
3. SUMMARY: 2-3 sentence summary of key findings
4. TOP_CONCERNS: List the 3 most critical issues found (or "None")
5. NEXT_STEPS: Recommended actions

Format as:
OVERALL_RISK: [1-10]
RECOMMENDATION: [APPROVE/REVIEW/REJECT]
SUMMARY: [brief summary]
TOP_CONCERNS: [concern1, concern2, concern3 or None]
NEXT_STEPS: [recommended actions]
"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            overall_risk = int(self._extract_field(text, "OVERALL_RISK", "5"))
            recommendation = self._extract_field(text, "RECOMMENDATION", "REVIEW")
            summary = self._extract_field(text, "SUMMARY", "Analysis completed")
            top_concerns_text = self._extract_field(text, "TOP_CONCERNS", "None")
            next_steps = self._extract_field(text, "NEXT_STEPS", "Manual review recommended")
            
            top_concerns = []
            if top_concerns_text.lower() != "none":
                top_concerns = [concern.strip() for concern in top_concerns_text.split(",")]
            
            return {
                "overall_risk": min(10, max(1, overall_risk)),
                "recommendation": recommendation.upper(),
                "summary": summary,
                "top_concerns": top_concerns,
                "next_steps": next_steps,
                "agent_details": [
                    {
                        "agent": r.agent_type,
                        "risk_score": r.risk_score,
                        "confidence": r.confidence,
                        "analysis": r.analysis,
                        "red_flags": r.red_flags
                    }
                    for r in agent_responses
                ],
                "statistics": {
                    "agents_consulted": len(agent_responses),
                    "average_risk": round(avg_risk, 1),
                    "average_confidence": round(avg_confidence, 1),
                    "total_red_flags": len(all_red_flags)
                }
            }
            
        except Exception as e:
            log.error(f"Final compilation failed: {e}")
            return {
                "overall_risk": int(avg_risk),
                "recommendation": "REVIEW",
                "summary": f"Analysis completed with {len(agent_responses)} agents. Compilation error: {str(e)}",
                "top_concerns": ["Compilation error"],
                "next_steps": "Manual review required due to system error",
                "agent_details": [],
                "statistics": {
                    "agents_consulted": len(agent_responses),
                    "average_risk": round(avg_risk, 1),
                    "average_confidence": round(avg_confidence, 1),
                    "total_red_flags": len(all_red_flags)
                }
            }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Invoice Fraud Detection")
    parser.add_argument("--invoice", help="Invoice data as string")
    parser.add_argument("--file", help="JSON file containing invoice data")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    if not args.invoice and not args.file:
        # Demo invoice for testing
        demo_invoice = """
        INVOICE #INV-2024-001
        
        From: QuickSupplies LLC
        123 Business Ave, Anytown, ST 12345
        
        To: ABC Corporation
        456 Corporate Blvd, Business City, ST 67890
        
        Date: 2024-03-15
        Due Date: 2024-04-15
        
        Items:
        - Office supplies (various): $2,847.50
        - Shipping & Handling: $150.00
        - Tax (8.5%): $254.94
        
        TOTAL: $3,252.44
        
        Payment Terms: Net 30
        Account: 1234-5678-9012
        """
        print("No invoice provided. Using demo invoice...")
        invoice_data = demo_invoice
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                data = json.load(f)
                invoice_data = json.dumps(data, indent=2)
        except Exception as e:
            log.error(f"Failed to read file {args.file}: {e}")
            return 1
    else:
        invoice_data = args.invoice
    
    try:
        detector = InvoiceFraudDetector()
        results = detector.analyze_invoice(invoice_data)
        
        # Print results
        print("\n" + "="*60)
        print("🕵️  INVOICE FRAUD DETECTION RESULTS")
        print("="*60)
        print(f"📊 Overall Risk: {results['overall_risk']}/10")
        print(f"🎯 Recommendation: {results['recommendation']}")
        print(f"📝 Summary: {results['summary']}")
        
        if results['top_concerns']:
            print(f"⚠️  Top Concerns:")
            for concern in results['top_concerns']:
                print(f"   • {concern}")
        
        print(f"📋 Next Steps: {results['next_steps']}")
        
        print(f"\n📈 Statistics:")
        stats = results['statistics']
        print(f"   Agents Consulted: {stats['agents_consulted']}")
        print(f"   Average Risk: {stats['average_risk']}/10")
        print(f"   Average Confidence: {stats['average_confidence']}/10")
        print(f"   Total Red Flags: {stats['total_red_flags']}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())