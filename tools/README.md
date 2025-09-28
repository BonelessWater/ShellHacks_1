# LLM Fraud Pattern Analyzer

This tool analyzes invoice data using multiple Large Language Models (LLMs) to identify and categorize different types of fraud patterns. It queries ChatGPT, Google Gemini, and Anthropic Claude to get diverse perspectives on fraud detection.

## 📁 Files Created

### In `tools/` directory:
- **`llm_fraud_analyzer.py`** - Main analysis script
- **`run_analysis.py`** - Simple runner script with setup checks
- **`requirements.txt`** - Python package dependencies
- **`.env.example`** - Template for API key configuration

### In `data/` directory:
- **`llm_fraud_analysis_results.json`** - Output file where LLM responses are stored

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd tools
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
# Edit .env file with your actual API keys
```

Required API keys:
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)  
- **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)

### 3. Run the Analysis
```bash
python run_analysis.py
```

Or run the main script directly:
```bash
python llm_fraud_analyzer.py --verbose
```

## 🔍 What It Analyzes

The tool sends 4 different types of prompts to each LLM:

### 1. **Pattern Identification**
- Identifies distinct fraud patterns in the data
- Provides specific indicators for each pattern
- Assigns risk severity levels
- Suggests algorithmic detection methods

### 2. **Detection Rules**
- Creates specific business rules for fraud detection
- Provides threshold values
- Suggests combination rules
- Includes false positive mitigation strategies

### 3. **Risk Scoring**
- Designs a 0-100 risk scoring system
- Assigns weights to different fraud indicators
- Provides scoring examples
- Recommends flagging thresholds

### 4. **Comparative Analysis**
- Compares fraudulent vs legitimate invoices
- Identifies key differences
- Provides statistical observations
- Offers fraud prevention recommendations

## 🎯 Fraud Patterns Detected

The system is designed to identify these fraud patterns:

- **Vendor Impersonation** - Mimicking legitimate companies
- **Price Inflation** - Excessive markup on services/products
- **Duplicate Billing** - Same services billed multiple times
- **Fake Vendor Addresses** - Non-existent or suspicious locations
- **Emergency Service Scams** - Fake urgent repairs with inflated costs
- **Hours Padding** - Inflated billable hours without verification
- **Service Bundling Fraud** - Combining services to hide inflated costs
- **Post-dated Invoice Fraud** - Backdated invoices submitted late
- **Tax Manipulation** - Incorrect or missing tax calculations
- **Payment Terms Abuse** - Unusually aggressive payment demands

## 📊 Output Format

Results are saved in `data/llm_fraud_analysis_results.json` with this structure:

```json
{
  "analysis_metadata": {
    "timestamp": "2024-09-27T...",
    "total_invoices_analyzed": 25,
    "fraudulent_invoices_count": 10,
    "legitimate_invoices_count": 15,
    "llm_responses_count": 12
  },
  "llm_responses": [
    {
      "model_name": "GPT-4",
      "prompt_type": "pattern_identification", 
      "response": "Detailed analysis...",
      "fraud_patterns_identified": [...],
      "confidence_scores": {...},
      "analysis_time_seconds": 3.2
    }
  ],
  "summary_analysis": {
    "models_used": ["GPT-4", "Gemini-Pro", "Claude-3-Sonnet"],
    "fraud_patterns_frequency": {...},
    "most_common_patterns": [...]
  }
}
```

## 🛠️ Technical Details

### Architecture
- **Async Processing**: Uses asyncio for concurrent LLM queries
- **Rate Limiting**: Built-in delays to respect API limits
- **Error Handling**: Graceful failure handling for each LLM
- **Modular Design**: Easy to add new LLMs or prompt types

### API Integration
- **OpenAI GPT-4**: Most detailed fraud analysis
- **Google Gemini Pro**: Alternative perspective and validation
- **Anthropic Claude-3**: Additional expert analysis

### Data Processing
- Automatically extracts fraud patterns from responses
- Calculates confidence scores based on response quality
- Aggregates patterns across all LLMs for comparison
- Tracks analysis performance and timing

## 📋 Requirements

### Python Packages
```
openai>=1.0.0
google-generativeai>=0.3.0
anthropic>=0.7.0
python-dotenv>=0.19.0
```

### API Access
- OpenAI API account with GPT-4 access
- Google AI API key for Gemini
- Anthropic API key for Claude

## 🔧 Configuration Options

### Command Line Arguments
```bash
python llm_fraud_analyzer.py --help

Options:
  --input     Path to invoice data JSON file (default: ../data/invoice_training_data.json)
  --output    Path to output results file (default: ../data/llm_fraud_analysis_results.json)  
  --verbose   Enable verbose logging
```

### Environment Variables
```bash
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here  
ANTHROPIC_API_KEY=your_key_here
MAX_REQUESTS_PER_MINUTE=10
REQUEST_DELAY_SECONDS=2
```

## 📈 Expected Output

After running the analysis, you'll get:

1. **Comprehensive fraud pattern identification** from multiple AI perspectives
2. **Actionable detection rules** that can be implemented in code
3. **Risk scoring methodologies** with specific numeric recommendations  
4. **Comparative insights** between fraudulent and legitimate invoices
5. **Performance metrics** showing analysis time and confidence scores

## 🚨 Important Notes

- **API Costs**: Running this analysis will consume API credits from each service
- **Rate Limits**: The script includes delays to respect API rate limits
- **Data Privacy**: Invoice data is sent to external AI services for analysis
- **Results Quality**: Output quality depends on the sophistication of each LLM

## 🔍 Next Steps

After getting the LLM analysis results, you can:

1. **Compare insights** from different AI models
2. **Implement detection rules** suggested by the LLMs
3. **Build a risk scoring system** based on the recommendations
4. **Create training data** for your own fraud detection models
5. **Validate patterns** against additional invoice datasets

## 🐛 Troubleshooting

### Common Issues:
- **API Key Errors**: Ensure all API keys are correctly set in environment variables
- **Rate Limiting**: If you hit rate limits, increase `REQUEST_DELAY_SECONDS`
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **Network Issues**: Check internet connection and API service status

### Debug Mode:
Run with `--verbose` flag to see detailed logging:
```bash
python llm_fraud_analyzer.py --verbose
```