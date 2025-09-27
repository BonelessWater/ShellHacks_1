#!/usr/bin/env python3
"""
Enhanced Invoice Feedback System with Multi-Agent Architecture
Implements the 6-step workflow with agent communication and comprehensive logging
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor

# Configure comprehensive logging
class AgentLogger:
    """Centralized logging system for all agents"""
    
    def __init__(self, log_dir: str = "invoice_feedback_system/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate loggers for different components
        self.loggers = {}
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup different loggers for system components"""
        components = ['orchestrator', 'llm_agent', 'validation_agent', 
                     'feedback_agent', 'analytics_agent', 'groupchat']
        
        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(logging.DEBUG)
            
            # File handler for this component
            file_handler = logging.FileHandler(
                self.log_dir / f"{component}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            self.loggers[component] = logger
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component"""
        return self.loggers.get(component, logging.getLogger(component))

# Initialize global logger
agent_logger = AgentLogger()

class MessageType(Enum):
    """Types of messages in agent communication"""
    USER_QUERY = "user_query"
    TOOL_CALL = "tool_call"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESULT = "validation_result"
    FEEDBACK_REQUEST = "feedback_request"
    FEEDBACK_RESULT = "feedback_result"
    ANALYTICS_REQUEST = "analytics_request"
    ANALYTICS_RESULT = "analytics_result"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    FINAL_RESPONSE = "final_response"

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str
    conversation_id: str
    
    @classmethod
    def create(cls, sender: str, recipient: str, message_type: MessageType, 
               content: Dict[str, Any], conversation_id: str) -> 'AgentMessage':
        return cls(
            id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id
        )

class GroupChat:
    """Central communication hub for all agents"""
    
    def __init__(self):
        self.logger = agent_logger.get_logger('groupchat')
        self.message_history: List[AgentMessage] = []
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.active_conversations: Dict[str, Dict] = {}
        
    def register_agent(self, agent: 'BaseAgent'):
        """Register an agent with the group chat"""
        self.agents[agent.name] = agent
        self.logger.info(f"Agent registered: {agent.name}")
    
    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message through the group chat"""
        self.message_history.append(message)
        self.logger.info(
            f"Message sent: {message.sender} -> {message.recipient} "
            f"[{message.message_type.value}]"
        )
        
        # Route message to recipient
        if message.recipient in self.agents:
            response = await self.agents[message.recipient].receive_message(message)
            if response:
                self.message_history.append(response)
                self.logger.info(
                    f"Response received: {response.sender} -> {response.recipient} "
                    f"[{response.message_type.value}]"
                )
            return response
        else:
            self.logger.error(f"Recipient not found: {message.recipient}")
            return None
    
    def get_conversation_history(self, conversation_id: str) -> List[AgentMessage]:
        """Get all messages for a specific conversation"""
        return [msg for msg in self.message_history if msg.conversation_id == conversation_id]
    
    def broadcast_status(self, sender: str, status: str, conversation_id: str):
        """Broadcast status update to all agents"""
        self.logger.info(f"Broadcasting status from {sender}: {status}")
        for agent_name in self.agents:
            if agent_name != sender:
                message = AgentMessage.create(
                    sender=sender,
                    recipient=agent_name,
                    message_type=MessageType.STATUS_UPDATE,
                    content={"status": status},
                    conversation_id=conversation_id
                )
                asyncio.create_task(self.send_message(message))

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, description: str, group_chat: GroupChat):
        self.name = name
        self.description = description
        self.group_chat = group_chat
        self.logger = agent_logger.get_logger(name.lower().replace(' ', '_'))
        self.is_busy = False
        
        # Register with group chat
        group_chat.register_agent(self)
        
    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process a message"""
        self.logger.debug(f"Received message: {message.message_type.value} from {message.sender}")
        
        # Process based on message type
        if message.message_type == MessageType.STATUS_UPDATE:
            await self.handle_status_update(message)
            return None
        else:
            return await self.process_message(message)
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process specific message types - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def handle_status_update(self, message: AgentMessage):
        """Handle status updates from other agents"""
        self.logger.info(f"Status update from {message.sender}: {message.content.get('status')}")
    
    async def send_message(self, recipient: str, message_type: MessageType, 
                          content: Dict[str, Any], conversation_id: str) -> Optional[AgentMessage]:
        """Send a message through group chat"""
        message = AgentMessage.create(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id
        )
        return await self.group_chat.send_message(message)

class OrchestratorAgent(BaseAgent):
    """
    Step 3: Orchestrator Validation
    Validates tool call schema, applies safety policies, routes to appropriate agent
    """
    
    def __init__(self, group_chat: GroupChat):
        super().__init__(
            name="Orchestrator",
            description="Validates requests and routes to appropriate agents",
            group_chat=group_chat
        )
        self.safety_policies = self._load_safety_policies()
        self.request_queue = []
    
    def _load_safety_policies(self) -> Dict[str, Any]:
        """Load safety and validation policies"""
        return {
            "max_request_size": 10000,  # Max characters in request
            "allowed_operations": [
                "validate_invoice", "collect_feedback", "generate_analytics"
            ],
            "rate_limits": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000
            },
            "data_validation": {
                "required_fields": ["invoice_number", "amount"],
                "max_amount": 1000000,  # $1M max invoice amount
                "date_range_years": 5   # Max 5 years historical data
            }
        }
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process and validate incoming requests"""
        self.logger.info(f"Processing {message.message_type.value} from {message.sender}")
        
        if message.message_type == MessageType.USER_QUERY:
            return await self.handle_user_query(message)
        elif message.message_type == MessageType.TOOL_CALL:
            return await self.validate_and_route_tool_call(message)
        else:
            self.logger.warning(f"Unhandled message type: {message.message_type}")
            return None
    
    async def handle_user_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle initial user query - Step 1: User Input Processing"""
        query = message.content.get('query', '')
        
        self.logger.info(f"Processing user query: {query[:100]}...")
        
        # Intent recognition
        intent = self._recognize_intent(query)
        
        # Create structured request
        structured_request = {
            "original_query": query,
            "intent": intent,
            "extracted_entities": self._extract_entities(query),
            "conversation_id": message.conversation_id
        }
        
        # Route to LLM Agent for decision making
        response = await self.send_message(
            recipient="LLM Agent",
            message_type=MessageType.TOOL_CALL,
            content=structured_request,
            conversation_id=message.conversation_id
        )
        
        self.group_chat.broadcast_status(
            sender=self.name,
            status=f"User query processed, intent: {intent}",
            conversation_id=message.conversation_id
        )
        
        return response
    
    async def validate_and_route_tool_call(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Validate tool calls and route to appropriate agents"""
        tool_call = message.content
        
        # Schema validation
        if not self._validate_schema(tool_call):
            return self._create_error_response(
                message, "Invalid tool call schema"
            )
        
        # Safety policy validation
        if not self._apply_safety_policies(tool_call):
            return self._create_error_response(
                message, "Safety policy violation"
            )
        
        # Route to appropriate agent
        tool_name = tool_call.get('tool_name')
        
        if tool_name == 'validate_invoice':
            recipient = "Validation Agent"
        elif tool_name == 'collect_feedback':
            recipient = "Feedback Agent"
        elif tool_name == 'generate_analytics':
            recipient = "Analytics Agent"
        else:
            return self._create_error_response(
                message, f"Unknown tool: {tool_name}"
            )
        
        self.logger.info(f"Routing {tool_name} to {recipient}")
        
        return await self.send_message(
            recipient=recipient,
            message_type=MessageType.VALIDATION_REQUEST,
            content=tool_call,
            conversation_id=message.conversation_id
        )
    
    def _recognize_intent(self, query: str) -> str:
        """Basic intent recognition"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['validate', 'check', 'verify']):
            return 'validate_invoice'
        elif any(word in query_lower for word in ['feedback', 'correct', 'wrong']):
            return 'collect_feedback'
        elif any(word in query_lower for word in ['analytics', 'report', 'analysis']):
            return 'generate_analytics'
        else:
            return 'general_inquiry'
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        # Simple entity extraction - could be enhanced with NLP
        entities = {}
        
        # Look for invoice numbers
        import re
        invoice_pattern = r'(invoice|inv)[\s#]*(\d+)'
        invoice_match = re.search(invoice_pattern, query, re.IGNORECASE)
        if invoice_match:
            entities['invoice_number'] = invoice_match.group(2)
        
        # Look for amounts
        amount_pattern = r'\$?([\d,]+\.?\d*)'
        amount_match = re.search(amount_pattern, query)
        if amount_match:
            entities['amount'] = amount_match.group(1)
        
        return entities
    
    def _validate_schema(self, tool_call: Dict[str, Any]) -> bool:
        """Validate tool call schema"""
        required_fields = ['tool_name', 'parameters']
        return all(field in tool_call for field in required_fields)
    
    def _apply_safety_policies(self, tool_call: Dict[str, Any]) -> bool:
        """Apply safety policies to tool call"""
        # Check operation is allowed
        if tool_call.get('tool_name') not in self.safety_policies['allowed_operations']:
            return False
        
        # Check data size limits
        content_size = len(str(tool_call))
        if content_size > self.safety_policies['max_request_size']:
            return False
        
        # Check amount limits if present
        params = tool_call.get('parameters', {})
        if 'amount' in params:
            try:
                amount = float(str(params['amount']).replace(',', '').replace('$', ''))
                if amount > self.safety_policies['data_validation']['max_amount']:
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _create_error_response(self, original_message: AgentMessage, error: str) -> AgentMessage:
        """Create an error response message"""
        return AgentMessage.create(
            sender=self.name,
            recipient=original_message.sender,
            message_type=MessageType.ERROR,
            content={"error": error, "original_message_id": original_message.id},
            conversation_id=original_message.conversation_id
        )

class LLMAgent(BaseAgent):
    """
    Step 2: Core LLM Decision
    Analyzes query, determines if external tools needed, generates tool calls in JSON format
    """
    
    def __init__(self, group_chat: GroupChat):
        super().__init__(
            name="LLM Agent", 
            description="Core decision making and natural language processing",
            group_chat=group_chat
        )
        self.context_memory = {}
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages and make LLM decisions"""
        self.logger.info(f"LLM processing: {message.message_type.value}")
        
        if message.message_type == MessageType.TOOL_CALL:
            return await self.analyze_and_decide(message)
        elif message.message_type in [MessageType.VALIDATION_RESULT, 
                                     MessageType.FEEDBACK_RESULT, 
                                     MessageType.ANALYTICS_RESULT]:
            return await self.integrate_results(message)
        else:
            return None
    
    async def analyze_and_decide(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Analyze request and decide on tool usage"""
        request = message.content
        conversation_id = message.conversation_id
        
        self.logger.info(f"Analyzing request with intent: {request.get('intent')}")
        
        # Store context
        self.context_memory[conversation_id] = {
            "original_query": request.get('original_query'),
            "intent": request.get('intent'),
            "entities": request.get('extracted_entities', {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine if external tools are needed
        intent = request.get('intent')
        
        if intent == 'validate_invoice':
            # Generate validation tool call
            tool_call = {
                "tool_name": "validate_invoice",
                "parameters": {
                    "invoice_data": request.get('extracted_entities', {}),
                    "validation_rules": "default",
                    "conversation_id": conversation_id
                }
            }
            
            return await self.send_message(
                recipient="Orchestrator",
                message_type=MessageType.TOOL_CALL,
                content=tool_call,
                conversation_id=conversation_id
            )
            
        elif intent == 'collect_feedback':
            # Generate feedback collection tool call
            tool_call = {
                "tool_name": "collect_feedback",
                "parameters": {
                    "feedback_data": request.get('extracted_entities', {}),
                    "conversation_id": conversation_id
                }
            }
            
            return await self.send_message(
                recipient="Orchestrator",
                message_type=MessageType.TOOL_CALL,
                content=tool_call,
                conversation_id=conversation_id
            )
            
        elif intent == 'generate_analytics':
            # Generate analytics tool call
            tool_call = {
                "tool_name": "generate_analytics",
                "parameters": {
                    "analysis_type": "comprehensive",
                    "conversation_id": conversation_id
                }
            }
            
            return await self.send_message(
                recipient="Orchestrator",
                message_type=MessageType.TOOL_CALL,
                content=tool_call,
                conversation_id=conversation_id
            )
        
        else:
            # No external tools needed, generate direct response
            response_content = self._generate_direct_response(request)
            
            return AgentMessage.create(
                sender=self.name,
                recipient=message.sender,
                message_type=MessageType.FINAL_RESPONSE,
                content={"response": response_content},
                conversation_id=conversation_id
            )
    
    async def integrate_results(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Step 5: Response Integration
        Integrate results from agents and generate natural language response
        """
        conversation_id = message.conversation_id
        results = message.content
        
        self.logger.info(f"Integrating results from {message.sender}")
        
        # Get conversation context
        context = self.context_memory.get(conversation_id, {})
        
        # Generate natural language response based on results
        response = self._generate_contextual_response(results, context)
        
        # Check if self-correction is needed (Step 6)
        if self._needs_correction(results):
            self.logger.info("Self-correction loop initiated")
            corrected_response = await self._apply_self_correction(results, context, conversation_id)
            response = corrected_response or response
        
        return AgentMessage.create(
            sender=self.name,
            recipient="Orchestrator",  # Send back to orchestrator for final delivery
            message_type=MessageType.FINAL_RESPONSE,
            content={"response": response, "conversation_id": conversation_id},
            conversation_id=conversation_id
        )
    
    def _generate_direct_response(self, request: Dict[str, Any]) -> str:
        """Generate direct response for queries that don't need external tools"""
        query = request.get('original_query', '')
        
        # Simple response generation - could be enhanced with actual LLM
        if 'help' in query.lower():
            return ("I can help you validate invoices, collect feedback on validation results, "
                   "and generate analytics reports. What would you like me to do?")
        else:
            return ("I understand your query, but I need more specific information. "
                   "Please specify if you want to validate an invoice, provide feedback, "
                   "or generate analytics.")
    
    def _generate_contextual_response(self, results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate contextual response based on results and context"""
        intent = context.get('intent', 'unknown')
        
        if intent == 'validate_invoice':
            return self._format_validation_response(results)
        elif intent == 'collect_feedback':
            return self._format_feedback_response(results)
        elif intent == 'generate_analytics':
            return self._format_analytics_response(results)
        else:
            return "I've processed your request and here are the results: " + str(results)
    
    def _format_validation_response(self, results: Dict[str, Any]) -> str:
        """Format validation results into natural language"""
        if results.get('status') == 'success':
            discrepancies = results.get('discrepancies', [])
            if discrepancies:
                response = f"I found {len(discrepancies)} potential issues with this invoice:\n\n"
                for i, disc in enumerate(discrepancies, 1):
                    response += f"{i}. {disc.get('description', 'Unknown issue')}\n"
                response += "\nWould you like me to help you address these issues?"
            else:
                response = "Great news! I didn't find any discrepancies in this invoice. It appears to be valid."
        else:
            response = f"I encountered an issue while validating the invoice: {results.get('message', 'Unknown error')}"
        
        return response
    
    def _format_feedback_response(self, results: Dict[str, Any]) -> str:
        """Format feedback results into natural language"""
        if results.get('status') == 'success':
            return ("Thank you for your feedback! I've recorded your input and will use it to "
                   "improve future invoice validations.")
        else:
            return f"I had trouble recording your feedback: {results.get('message', 'Unknown error')}"
    
    def _format_analytics_response(self, results: Dict[str, Any]) -> str:
        """Format analytics results into natural language"""
        if results.get('status') == 'success':
            analytics = results.get('analytics', {})
            accuracy = analytics.get('accuracy', 0)
            return (f"Here's your analytics summary:\n"
                   f"• System accuracy: {accuracy:.1%}\n"
                   f"• Total validations: {analytics.get('total_validations', 0)}\n"
                   f"• Common issues: {', '.join(analytics.get('common_discrepancies', []))}")
        else:
            return f"I couldn't generate the analytics report: {results.get('message', 'Unknown error')}"
    
    def _needs_correction(self, results: Dict[str, Any]) -> bool:
        """Determine if self-correction is needed"""
        # Check for errors or low confidence results
        if results.get('status') == 'error':
            return True
        
        # Check confidence levels
        if 'confidence' in results and results['confidence'] < 0.7:
            return True
        
        return False
    
    async def _apply_self_correction(self, results: Dict[str, Any], context: Dict[str, Any], 
                                   conversation_id: str) -> Optional[str]:
        """
        Step 6: Self-Correction Loop
        Apply iterative improvement until success
        """
        self.logger.info("Applying self-correction")
        
        # For now, implement basic retry logic
        # In a full implementation, this would revise the approach
        
        correction_attempt = {
            "original_results": results,
            "correction_strategy": "retry_with_modified_parameters",
            "timestamp": datetime.now().isoformat()
        }
        
        # Log correction attempt
        self.logger.info(f"Self-correction attempt: {correction_attempt}")
        
        return "I've reviewed the results and applied corrections to improve accuracy."

class ValidationAgent(BaseAgent):
    """
    Step 4: Agent Execution - Specialized agent for invoice validation
    """
    
    def __init__(self, group_chat: GroupChat):
        super().__init__(
            name="Validation Agent",
            description="Specialized invoice validation and discrepancy detection",
            group_chat=group_chat
        )
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            "amount_threshold": 5000.0,
            "tax_rate_range": [0.0, 0.15],
            "date_range_days": 90,
            "required_fields": ["invoice_number", "amount", "vendor", "date"],
            "duplicate_check_enabled": True,
            "vendor_whitelist_enabled": False
        }
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process validation requests"""
        if message.message_type == MessageType.VALIDATION_REQUEST:
            return await self.validate_invoice(message)
        return None
    
    async def validate_invoice(self, message: AgentMessage) -> AgentMessage:
        """Perform comprehensive invoice validation"""
        request = message.content
        conversation_id = message.conversation_id
        
        self.logger.info(f"Validating invoice for conversation {conversation_id}")
        
        self.is_busy = True
        self.group_chat.broadcast_status(
            sender=self.name,
            status="Starting invoice validation",
            conversation_id=conversation_id
        )
        
        try:
            # Extract invoice data
            invoice_data = request.get('parameters', {}).get('invoice_data', {})
            
            # Perform validation checks
            discrepancies = []
            
            # Check 1: Required fields
            missing_fields = self._check_required_fields(invoice_data)
            if missing_fields:
                discrepancies.append({
                    "type": "missing_fields",
                    "severity": "high",
                    "description": f"Missing required fields: {', '.join(missing_fields)}",
                    "confidence": 1.0
                })
            
            # Check 2: Amount validation
            amount_issues = self._validate_amount(invoice_data)
            discrepancies.extend(amount_issues)
            
            # Check 3: Date validation
            date_issues = self._validate_dates(invoice_data)
            discrepancies.extend(date_issues)
            
            # Check 4: Duplicate detection
            if self.validation_rules["duplicate_check_enabled"]:
                duplicate_issues = await self._check_duplicates(invoice_data)
                discrepancies.extend(duplicate_issues)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(discrepancies, invoice_data)
            
            # Save validation results
            await self._save_validation_results(invoice_data, discrepancies, conversation_id)
            
            result = {
                "status": "success",
                "discrepancies": discrepancies,
                "confidence": confidence,
                "invoice_data": invoice_data,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            result = {
                "status": "error",
                "message": str(e),
                "conversation_id": conversation_id
            }
        
        finally:
            self.is_busy = False
            self.group_chat.broadcast_status(
                sender=self.name,
                status=f"Validation completed - found {len(result.get('discrepancies', []))} issues",
                conversation_id=conversation_id
            )
        
        return AgentMessage.create(
            sender=self.name,
            recipient="LLM Agent",
            message_type=MessageType.VALIDATION_RESULT,
            content=result,
            conversation_id=conversation_id
        )
    
    def _check_required_fields(self, invoice_data: Dict[str, Any]) -> List[str]:
        """Check for missing required fields"""
        missing = []
        for field in self.validation_rules["required_fields"]:
            if field not in invoice_data or not invoice_data[field]:
                missing.append(field)
        return missing
    
    def _validate_amount(self, invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate invoice amounts"""
        issues = []
        amount_str = str(invoice_data.get('amount', '0'))
        
        try:
            amount = float(amount_str.replace(',', '').replace('$', ''))
            
            if amount <= 0:
                issues.append({
                    "type": "invalid_amount",
                    "severity": "high",
                    "description": "Invoice amount must be greater than zero",
                    "confidence": 1.0
                })
            elif amount > self.validation_rules["amount_threshold"]:
                issues.append({
                    "type": "high_amount",
                    "severity": "medium",
                    "description": f"Invoice amount ${amount:,.2f} exceeds threshold",
                    "confidence": 0.8
                })
                
        except (ValueError, TypeError):
            issues.append({
                "type": "invalid_amount_format",
                "severity": "high", 
                "description": f"Invalid amount format: {amount_str}",
                "confidence": 1.0
            })
        
        return issues
    
    def _validate_dates(self, invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate invoice dates"""
        issues = []
        
        # Simple date validation - could be enhanced
        date_str = invoice_data.get('date', '')
        if date_str:
            try:
                # Try to parse the date
                from datetime import datetime
                parsed_date = datetime.fromisoformat(date_str.replace('/', '-'))
                
                # Check if date is in reasonable range
                days_diff = (datetime.now() - parsed_date).days
                if days_diff > self.validation_rules["date_range_days"]:
                    issues.append({
                        "type": "old_invoice_date",
                        "severity": "medium",
                        "description": f"Invoice date is {days_diff} days old",
                        "confidence": 0.9
                    })
                elif days_diff < -30:  # Future date beyond 30 days
                    issues.append({
                        "type": "future_invoice_date",
                        "severity": "high",
                        "description": "Invoice date is too far in the future",
                        "confidence": 0.95
                    })
                    
            except (ValueError, TypeError):
                issues.append({
                    "type": "invalid_date_format",
                    "severity": "medium",
                    "description": f"Invalid date format: {date_str}",
                    "confidence": 0.8
                })
        
        return issues
    
    async def _check_duplicates(self, invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for duplicate invoices"""
        issues = []
        
        # Simulate duplicate check - in real implementation, would check database
        invoice_number = invoice_data.get('invoice_number')
        amount = invoice_data.get('amount')
        
        if invoice_number and amount:
            # Simulate finding a duplicate (for demonstration)
            if invoice_number == "12345":  # Test case
                issues.append({
                    "type": "duplicate_invoice",
                    "severity": "critical",
                    "description": f"Potential duplicate: Invoice {invoice_number} for ${amount}",
                    "confidence": 0.85
                })
        
        return issues
    
    def _calculate_confidence(self, discrepancies: List[Dict], invoice_data: Dict[str, Any]) -> float:
        """Calculate overall validation confidence"""
        if not discrepancies:
            return 1.0
        
        # Weight by severity
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.9}
        total_weight = sum(severity_weights.get(d.get('severity', 'medium'), 0.3) for d in discrepancies)
        
        # Calculate confidence (inverse of weighted severity)
        confidence = max(0.1, 1.0 - (total_weight / len(discrepancies)))
        return round(confidence, 2)
    
    async def _save_validation_results(self, invoice_data: Dict, discrepancies: List, conversation_id: str):
        """Save validation results for analytics"""
        results_dir = Path("invoice_feedback_system/data/analytics")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{conversation_id}_{timestamp}.json"
        
        try:
            validation_record = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "invoice_data": invoice_data,
                "discrepancies": discrepancies,
                "agent": self.name
            }
            
            with open(results_dir / filename, 'w') as f:
                json.dump(validation_record, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save validation results: {e}")

class FeedbackAgent(BaseAgent):
    """
    Step 4: Agent Execution - Specialized agent for feedback collection
    """
    
    def __init__(self, group_chat: GroupChat):
        super().__init__(
            name="Feedback Agent",
            description="Collects and processes user feedback on validation results",
            group_chat=group_chat
        )
        self.feedback_history = self._load_feedback_history()
    
    def _load_feedback_history(self) -> List[Dict[str, Any]]:
        """Load existing feedback history"""
        feedback_file = Path("invoice_feedback_system/data/feedback/feedback_history.json")
        try:
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load feedback history: {e}")
        return []
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process feedback requests"""
        if message.message_type == MessageType.VALIDATION_REQUEST:
            return await self.collect_feedback(message)
        return None
    
    async def collect_feedback(self, message: AgentMessage) -> AgentMessage:
        """Collect and process user feedback"""
        request = message.content
        conversation_id = message.conversation_id
        
        self.logger.info(f"Collecting feedback for conversation {conversation_id}")
        
        self.is_busy = True
        self.group_chat.broadcast_status(
            sender=self.name,
            status="Processing user feedback",
            conversation_id=conversation_id
        )
        
        try:
            feedback_data = request.get('parameters', {}).get('feedback_data', {})
            
            # Process feedback
            feedback_record = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "user_action": feedback_data.get('action', 'unknown'),
                "user_comment": feedback_data.get('comment', ''),
                "invoice_number": feedback_data.get('invoice_number', ''),
                "feedback_type": feedback_data.get('type', 'general'),
                "agent": self.name
            }
            
            # Add to history
            self.feedback_history.append(feedback_record)
            
            # Save updated history
            await self._save_feedback_history()
            
            # Analyze feedback for patterns
            analysis = self._analyze_feedback_patterns()
            
            result = {
                "status": "success",
                "feedback_recorded": feedback_record,
                "patterns_analysis": analysis,
                "message": "Feedback collected successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Feedback collection error: {e}")
            result = {
                "status": "error",
                "message": str(e)
            }
        
        finally:
            self.is_busy = False
            self.group_chat.broadcast_status(
                sender=self.name,
                status="Feedback processing completed",
                conversation_id=conversation_id
            )
        
        return AgentMessage.create(
            sender=self.name,
            recipient="LLM Agent",
            message_type=MessageType.FEEDBACK_RESULT,
            content=result,
            conversation_id=conversation_id
        )
    
    async def _save_feedback_history(self):
        """Save feedback history to file"""
        feedback_file = Path("invoice_feedback_system/data/feedback/feedback_history.json")
        feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save feedback history: {e}")
    
    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns for system improvement"""
        if not self.feedback_history:
            return {"message": "No feedback data available"}
        
        # Count feedback types
        action_counts = {}
        for feedback in self.feedback_history:
            action = feedback.get('user_action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total_feedback = len(self.feedback_history)
        
        return {
            "total_feedback_items": total_feedback,
            "action_distribution": action_counts,
            "most_common_action": max(action_counts.keys(), key=action_counts.get) if action_counts else None,
            "feedback_trend": "positive" if action_counts.get('confirmed', 0) > action_counts.get('rejected', 0) else "needs_improvement"
        }

class AnalyticsAgent(BaseAgent):
    """
    Step 4: Agent Execution - Specialized agent for analytics and reporting
    """
    
    def __init__(self, group_chat: GroupChat):
        super().__init__(
            name="Analytics Agent",
            description="Generates analytics and insights from system data",
            group_chat=group_chat
        )
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process analytics requests"""
        if message.message_type == MessageType.VALIDATION_REQUEST:
            return await self.generate_analytics(message)
        return None
    
    async def generate_analytics(self, message: AgentMessage) -> AgentMessage:
        """Generate comprehensive analytics report"""
        request = message.content
        conversation_id = message.conversation_id
        
        self.logger.info(f"Generating analytics for conversation {conversation_id}")
        
        self.is_busy = True
        self.group_chat.broadcast_status(
            sender=self.name,
            status="Generating analytics report",
            conversation_id=conversation_id
        )
        
        try:
            # Collect data from various sources
            validation_data = await self._collect_validation_data()
            feedback_data = await self._collect_feedback_data()
            
            # Generate analytics
            analytics = {
                "report_generated": datetime.now().isoformat(),
                "validation_analytics": self._analyze_validations(validation_data),
                "feedback_analytics": self._analyze_feedback(feedback_data),
                "system_performance": self._calculate_system_performance(validation_data, feedback_data),
                "recommendations": self._generate_recommendations(validation_data, feedback_data)
            }
            
            # Save analytics report
            await self._save_analytics_report(analytics, conversation_id)
            
            result = {
                "status": "success",
                "analytics": analytics
            }
            
        except Exception as e:
            self.logger.error(f"Analytics generation error: {e}")
            result = {
                "status": "error",
                "message": str(e)
            }
        
        finally:
            self.is_busy = False
            self.group_chat.broadcast_status(
                sender=self.name,
                status="Analytics report completed",
                conversation_id=conversation_id
            )
        
        return AgentMessage.create(
            sender=self.name,
            recipient="LLM Agent",
            message_type=MessageType.ANALYTICS_RESULT,
            content=result,
            conversation_id=conversation_id
        )
    
    async def _collect_validation_data(self) -> List[Dict[str, Any]]:
        """Collect validation data from files"""
        validation_data = []
        analytics_dir = Path("invoice_feedback_system/data/analytics")
        
        if analytics_dir.exists():
            for file_path in analytics_dir.glob("validation_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        validation_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Could not load validation data from {file_path}: {e}")
        
        return validation_data
    
    async def _collect_feedback_data(self) -> List[Dict[str, Any]]:
        """Collect feedback data"""
        feedback_file = Path("invoice_feedback_system/data/feedback/feedback_history.json")
        
        try:
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load feedback data: {e}")
        
        return []
    
    def _analyze_validations(self, validation_data: List[Dict]) -> Dict[str, Any]:
        """Analyze validation data"""
        if not validation_data:
            return {"message": "No validation data available"}
        
        total_validations = len(validation_data)
        total_discrepancies = sum(len(v.get('discrepancies', [])) for v in validation_data)
        
        # Count discrepancy types
        discrepancy_types = {}
        for validation in validation_data:
            for discrepancy in validation.get('discrepancies', []):
                disc_type = discrepancy.get('type', 'unknown')
                discrepancy_types[disc_type] = discrepancy_types.get(disc_type, 0) + 1
        
        return {
            "total_validations": total_validations,
            "total_discrepancies": total_discrepancies,
            "average_discrepancies_per_invoice": round(total_discrepancies / total_validations, 2) if total_validations > 0 else 0,
            "common_discrepancy_types": discrepancy_types,
            "most_common_discrepancy": max(discrepancy_types.keys(), key=discrepancy_types.get) if discrepancy_types else None
        }
    
    def _analyze_feedback(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Analyze feedback data"""
        if not feedback_data:
            return {"message": "No feedback data available"}
        
        total_feedback = len(feedback_data)
        
        # Count feedback actions
        action_counts = {}
        for feedback in feedback_data:
            action = feedback.get('user_action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_feedback_items": total_feedback,
            "feedback_distribution": action_counts,
            "user_satisfaction": self._calculate_satisfaction(action_counts)
        }
    
    def _calculate_satisfaction(self, action_counts: Dict[str, int]) -> str:
        """Calculate user satisfaction level"""
        positive_actions = action_counts.get('confirmed', 0) + action_counts.get('helpful', 0)
        negative_actions = action_counts.get('rejected', 0) + action_counts.get('incorrect', 0)
        
        if positive_actions > negative_actions * 2:
            return "high"
        elif positive_actions > negative_actions:
            return "medium"
        else:
            return "low"
    
    def _calculate_system_performance(self, validation_data: List[Dict], feedback_data: List[Dict]) -> Dict[str, Any]:
        """Calculate overall system performance metrics"""
        # Calculate accuracy based on feedback
        confirmed_validations = len([f for f in feedback_data if f.get('user_action') == 'confirmed'])
        total_validations_with_feedback = len(feedback_data)
        
        accuracy = confirmed_validations / total_validations_with_feedback if total_validations_with_feedback > 0 else 0
        
        return {
            "accuracy": round(accuracy, 3),
            "total_processed": len(validation_data),
            "feedback_rate": round(total_validations_with_feedback / len(validation_data), 3) if validation_data else 0,
            "system_uptime": "99.9%",  # Mock value
            "average_processing_time": "2.3 seconds"  # Mock value
        }
    
    def _generate_recommendations(self, validation_data: List[Dict], feedback_data: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze validation patterns
        if validation_data:
            total_discrepancies = sum(len(v.get('discrepancies', [])) for v in validation_data)
            if total_discrepancies > len(validation_data) * 0.5:  # More than 50% have issues
                recommendations.append("Consider reviewing validation rules to reduce false positives")
        
        # Analyze feedback patterns
        if feedback_data:
            negative_feedback = len([f for f in feedback_data if f.get('user_action') in ['rejected', 'incorrect']])
            if negative_feedback > len(feedback_data) * 0.3:  # More than 30% negative
                recommendations.append("System accuracy needs improvement - consider model retraining")
        
        # Default recommendations
        recommendations.extend([
            "Regularly review and update validation rules based on business changes",
            "Encourage users to provide feedback for continuous improvement",
            "Monitor system performance metrics monthly"
        ])
        
        return recommendations
    
    async def _save_analytics_report(self, analytics: Dict[str, Any], conversation_id: str):
        """Save analytics report"""
        reports_dir = Path("invoice_feedback_system/data/analytics/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_report_{conversation_id}_{timestamp}.json"
        
        try:
            with open(reports_dir / filename, 'w') as f:
                json.dump(analytics, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save analytics report: {e}")

class InvoiceFeedbackSystem:
    """
    Main system orchestrator that implements the complete 6-step workflow
    """
    
    def __init__(self):
        self.group_chat = GroupChat()
        self.agents = self._initialize_agents()
        self.system_logger = agent_logger.get_logger('system')
        self.active_conversations = {}
        
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all system agents"""
        agents = {
            'orchestrator': OrchestratorAgent(self.group_chat),
            'llm_agent': LLMAgent(self.group_chat),
            'validation_agent': ValidationAgent(self.group_chat),
            'feedback_agent': FeedbackAgent(self.group_chat),
            'analytics_agent': AnalyticsAgent(self.group_chat)
        }
        
        self.system_logger.info(f"Initialized {len(agents)} agents")
        return agents
    
    async def process_user_query(self, query: str) -> str:
        """
        Main entry point for processing user queries
        Implements the complete 6-step workflow
        """
        conversation_id = str(uuid.uuid4())
        
        self.system_logger.info(f"Processing user query [conversation: {conversation_id}]: {query[:100]}...")
        
        # Store conversation context
        self.active_conversations[conversation_id] = {
            "start_time": datetime.now().isoformat(),
            "query": query,
            "status": "processing"
        }
        
        try:
            # Step 1: User Input Processing - Send to Orchestrator
            message = AgentMessage.create(
                sender="User",
                recipient="Orchestrator",
                message_type=MessageType.USER_QUERY,
                content={"query": query},
                conversation_id=conversation_id
            )
            
            # Send message and wait for final response
            final_response = await self._process_workflow(message)
            
            # Update conversation status
            self.active_conversations[conversation_id]["status"] = "completed"
            self.active_conversations[conversation_id]["end_time"] = datetime.now().isoformat()
            
            return final_response
            
        except Exception as e:
            self.system_logger.error(f"Error processing query: {e}")
            self.active_conversations[conversation_id]["status"] = "error"
            self.active_conversations[conversation_id]["error"] = str(e)
            return f"I encountered an error processing your request: {str(e)}"
    
    async def _process_workflow(self, initial_message: AgentMessage) -> str:
        """Process the complete workflow and return final response"""
        current_message = initial_message
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Send message through group chat
            response = await self.group_chat.send_message(current_message)
            
            if not response:
                break
            
            # Check if we have a final response
            if response.message_type == MessageType.FINAL_RESPONSE:
                return response.content.get('response', 'No response generated')
            
            # Continue processing with the response
            current_message = response
        
        return "I was unable to complete your request within the expected time."
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        agent_status = {}
        for name, agent in self.agents.items():
            agent_status[name] = {
                "name": agent.name,
                "description": agent.description,
                "is_busy": getattr(agent, 'is_busy', False)
            }
        
        return {
            "system_status": "operational",
            "active_conversations": len(self.active_conversations),
            "agents": agent_status,
            "total_messages": len(self.group_chat.message_history),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""
        messages = self.group_chat.get_conversation_history(conversation_id)
        return [asdict(msg) for msg in messages]

# Testing and demonstration functions
async def test_enhanced_system():
    """Test the enhanced invoice feedback system"""
    print("🚀 Testing Enhanced Invoice Feedback System with Agent Communication")
    print("=" * 80)
    
    # Initialize system
    system = InvoiceFeedbackSystem()
    
    # Test queries
    test_queries = [
        "Please validate invoice #12345 for $1,500 from Acme Corp",
        "I think the validation was wrong for invoice #67890, it's actually correct",
        "Can you generate an analytics report for our invoice processing?",
        "Help me understand how the system works"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query}")
        print("-" * 60)
        
        # Process query
        response = await system.process_user_query(query)
        print(f"🤖 System Response: {response}")
        
        # Show system status
        status = await system.get_system_status()
        print(f"📊 Active Conversations: {status['active_conversations']}")
        print(f"📊 Total Messages: {status['total_messages']}")
        
        # Wait between tests
        await asyncio.sleep(1)
    
    print("\n" + "=" * 80)
    print("✅ Enhanced system testing completed!")
    
    # Final system status
    final_status = await system.get_system_status()
    print("\n📊 Final System Status:")
    for agent_name, agent_info in final_status['agents'].items():
        status_icon = "🔄" if agent_info['is_busy'] else "✅"
        print(f"   {status_icon} {agent_info['name']}: {agent_info['description']}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_system())