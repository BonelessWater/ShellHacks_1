#!/usr/bin/env python3
"""
Comprehensive Multi-Gemini Model Fraud Analyzer
Tests and implements all available Gemini models with fraud detection data
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import logging

class ComprehensiveGeminiAnalyzer:
    """Complete Gemini model implementation with fraud data analysis"""
    
    def __init__(self):
        """Initialize the comprehensive analyzer"""
        load_dotenv('.env')
        self.setup_logging()
        self.discover_available_models()
        self.fraud_data = None
        
    def setup_logging(self):
        """Setup logging for detailed tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('../data/gemini_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def discover_available_models(self):
        """Discover all available Gemini models"""
        print("🔍 DISCOVERING ALL AVAILABLE GEMINI MODELS")
        print("=" * 55)
        
        try:
            import google.generativeai as genai
            
            # Test both API keys
            api_keys = {
                'primary': os.getenv('GOOGLE_API_KEY'),
                'backup': os.getenv('GOOGLE_API_KEY_ILAN')
            }
            
            self.available_models = {}
            self.working_api_key = None
            
            for key_name, api_key in api_keys.items():
                if not api_key:
                    continue
                    
                print(f"\n🔑 Testing {key_name} key: {api_key[:20]}...")
                
                try:
                    genai.configure(api_key=api_key)
                    
                    # Get ALL available models
                    models = list(genai.list_models())
                    generation_models = [
                        model for model in models 
                        if 'generateContent' in model.supported_generation_methods
                    ]
                    
                    print(f"   📋 Found {len(generation_models)} generation models")
                    
                    # Categorize models by generation
                    model_categories = {
                        'Gemini 1.0 (Original)': [],
                        'Gemini 1.5 (Enhanced)': [],
                        'Gemini 2.0 (Latest)': [],
                        'Gemini Pro (Advanced)': [],
                        'Gemini Flash (Fast)': [],
                        'Other/Experimental': []
                    }
                    
                    for model in generation_models:
                        model_name = model.name
                        
                        if 'gemini-pro' in model_name and '1.5' not in model_name and '2.0' not in model_name:
                            model_categories['Gemini 1.0 (Original)'].append(model_name)
                        elif '1.5' in model_name and 'pro' in model_name:
                            model_categories['Gemini 1.5 (Enhanced)'].append(model_name)
                        elif '1.5' in model_name and 'flash' in model_name:
                            model_categories['Gemini Flash (Fast)'].append(model_name)
                        elif '2.0' in model_name:
                            model_categories['Gemini 2.0 (Latest)'].append(model_name)
                        elif 'pro' in model_name:
                            model_categories['Gemini Pro (Advanced)'].append(model_name)
                        elif 'flash' in model_name:
                            model_categories['Gemini Flash (Fast)'].append(model_name)
                        else:
                            model_categories['Other/Experimental'].append(model_name)
                    
                    # Display found models
                    total_found = 0
                    for category, models_list in model_categories.items():
                        if models_list:
                            print(f"\n   📂 {category} ({len(models_list)} models):")
                            for model in models_list:
                                print(f"      • {model}")
                                total_found += 1
                    
                    if total_found > 0:
                        self.available_models = {cat: models for cat, models in model_categories.items() if models}
                        self.working_api_key = api_key
                        self.genai = genai
                        print(f"\n✅ SUCCESS: Using {key_name} key with {total_found} models")
                        break
                    
                except Exception as e:
                    print(f"   ❌ Failed with {key_name} key: {str(e)[:100]}")
                    continue
            
            if not self.available_models:
                print("\n❌ No working Gemini models found!")
                self.available_models = {}
                
        except ImportError:
            print("❌ Google Generative AI library not installed")
            self.available_models = {}
    
    def load_fraud_data(self):
        """Load BigQuery fraud samples"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_data = data['datasets']
                
                # Create analysis-ready data samples
                self.fraud_samples = {}
                for dataset_name, records in self.fraud_data.items():
                    fraud_cases = [r for r in records if r.get('isFraud', r.get('Class', 0)) == 1]
                    normal_cases = [r for r in records if r.get('isFraud', r.get('Class', 0)) == 0]
                    
                    self.fraud_samples[dataset_name] = {
                        'fraud_cases': fraud_cases[:5],  # Top 5 fraud cases
                        'normal_cases': normal_cases[:3],  # Top 3 normal cases
                        'total_fraud': len(fraud_cases),
                        'total_normal': len(normal_cases)
                    }
                
                total_samples = sum(len(records) for records in self.fraud_data.values())
                print(f"📊 Loaded fraud data: {total_samples} samples across {len(self.fraud_data)} datasets")
                return True
                
        except FileNotFoundError:
            print("❌ Fraud data not found. Run bigquery_fraud_analyzer_fixed.py first.")
            return False
    
    def create_fraud_analysis_prompts(self):
        """Create different analysis prompts for different model types"""
        return {
            'basic_analysis': """
            You are a fraud detection expert. Analyze the provided transaction data and identify fraud patterns.
            
            Data: {data_sample}
            
            Tasks:
            1. Identify the top 3 fraud indicators
            2. Explain what makes transactions fraudulent
            3. Suggest detection rules
            
            Be specific and actionable.
            """,
            
            'detailed_analysis': """
            You are a senior fraud analyst examining financial transaction data for a fraud detection system.
            
            Dataset Context:
            {dataset_context}
            
            Sample Data:
            {data_sample}
            
            Analysis Requirements:
            1. **Pattern Recognition**: What patterns distinguish fraud from legitimate transactions?
            2. **Risk Factors**: Which transaction features are most predictive of fraud?
            3. **Detection Strategy**: How would you implement these insights in a real-time system?
            4. **False Positive Mitigation**: How to avoid flagging legitimate transactions?
            5. **Business Impact**: What's the potential ROI of implementing these patterns?
            
            Provide detailed, data-driven insights suitable for production implementation.
            """,
            
            'comparative_analysis': """
            Compare fraud patterns across multiple datasets:
            
            {multi_dataset_info}
            
            Analysis Focus:
            1. Cross-dataset patterns that appear consistently
            2. Dataset-specific fraud characteristics
            3. Universal fraud detection rules vs dataset-specific rules
            4. Model generalization potential
            
            Provide insights for building a robust, generalizable fraud detection system.
            """
        }
    
    async def test_model_capability(self, model_name, test_type='basic'):
        """Test a specific Gemini model's capability"""
        try:
            model = self.genai.GenerativeModel(model_name)
            
            # Simple capability test
            test_prompt = "Explain fraud detection in 2 sentences."
            
            start_time = time.time()
            response = model.generate_content(
                test_prompt,
                generation_config=self.genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.1
                )
            )
            end_time = time.time()
            
            if response and response.text:
                return {
                    'status': 'working',
                    'response_time': round(end_time - start_time, 2),
                    'response_length': len(response.text),
                    'response_sample': response.text[:100] + "..." if len(response.text) > 100 else response.text
                }
            else:
                return {'status': 'no_response', 'error': 'Empty response'}
                
        except Exception as e:
            error_msg = str(e)
            if 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                return {'status': 'quota_exceeded', 'error': 'Rate/quota limit hit'}
            elif 'not found' in error_msg.lower() or '404' in error_msg:
                return {'status': 'not_found', 'error': 'Model not available'}
            else:
                return {'status': 'error', 'error': error_msg[:100]}
    
    async def analyze_with_model(self, model_name, analysis_type='basic'):
        """Run fraud analysis with a specific model"""
        if not self.fraud_data:
            return {'error': 'No fraud data loaded'}
        
        try:
            model = self.genai.GenerativeModel(model_name)
            prompts = self.create_fraud_analysis_prompts()
            
            # Prepare data based on analysis type
            if analysis_type == 'basic':
                # Use IEEE CIS data for basic analysis
                ieee_data = self.fraud_samples.get('ieee_cis_fraud', {})
                data_sample = {
                    'fraud_examples': ieee_data.get('fraud_cases', [])[:3],
                    'normal_examples': ieee_data.get('normal_cases', [])[:2]
                }
                prompt = prompts['basic_analysis'].format(data_sample=json.dumps(data_sample, indent=2, default=str)[:2000])
                
            elif analysis_type == 'detailed':
                # Use multiple datasets for detailed analysis
                dataset_context = {name: {'fraud_count': data['total_fraud'], 'normal_count': data['total_normal']} 
                                 for name, data in self.fraud_samples.items()}
                
                # Use PaySim data for detailed analysis (has clear fraud patterns)
                paysim_data = self.fraud_samples.get('paysim_fraud', {})
                data_sample = {
                    'fraud_transactions': paysim_data.get('fraud_cases', [])[:3],
                    'normal_transactions': paysim_data.get('normal_cases', [])[:2]
                }
                
                prompt = prompts['detailed_analysis'].format(
                    dataset_context=json.dumps(dataset_context, indent=2),
                    data_sample=json.dumps(data_sample, indent=2, default=str)[:2000]
                )
                
            else:  # comparative
                multi_dataset_info = {
                    name: {
                        'fraud_sample': data['fraud_cases'][:2],
                        'normal_sample': data['normal_cases'][:1],
                        'fraud_total': data['total_fraud']
                    } for name, data in self.fraud_samples.items()
                }
                prompt = prompts['comparative_analysis'].format(
                    multi_dataset_info=json.dumps(multi_dataset_info, indent=2, default=str)[:3000]
                )
            
            # Generate analysis
            start_time = time.time()
            response = model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.2
                )
            )
            end_time = time.time()
            
            if response and response.text:
                return {
                    'model': model_name,
                    'analysis_type': analysis_type,
                    'status': 'success',
                    'analysis': response.text,
                    'analysis_length': len(response.text),
                    'response_time': round(end_time - start_time, 2),
                    'timestamp': datetime.now().isoformat(),
                    'data_used': list(self.fraud_samples.keys())
                }
            else:
                return {
                    'model': model_name,
                    'analysis_type': analysis_type,
                    'status': 'no_response',
                    'error': 'Empty response'
                }
                
        except Exception as e:
            return {
                'model': model_name,
                'analysis_type': analysis_type,
                'status': 'error',
                'error': str(e)[:200],
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_comprehensive_analysis(self):
        """Run comprehensive analysis across all available models"""
        print("\n🚀 COMPREHENSIVE MULTI-GEMINI FRAUD ANALYSIS")
        print("=" * 55)
        
        if not self.available_models:
            print("❌ No models available for analysis")
            return None
        
        # Load fraud data
        if not self.load_fraud_data():
            return None
        
        # First, test model capabilities
        print("\n🧪 TESTING MODEL CAPABILITIES...")
        model_capabilities = {}
        
        all_models = []
        for category, models in self.available_models.items():
            all_models.extend(models)
        
        print(f"Testing {len(all_models)} models...")
        
        for model_name in all_models:
            print(f"   Testing {model_name}...")
            capability = await self.test_model_capability(model_name)
            model_capabilities[model_name] = capability
            
            if capability['status'] == 'working':
                print(f"      ✅ Working: {capability['response_time']}s, {capability['response_length']} chars")
            else:
                print(f"      ❌ {capability['status']}: {capability.get('error', 'Unknown error')}")
            
            await asyncio.sleep(3)  # Rate limiting
        
        # Filter working models
        working_models = [name for name, cap in model_capabilities.items() if cap['status'] == 'working']
        
        if not working_models:
            print("\n❌ No working models found!")
            return {'error': 'No working models', 'capabilities': model_capabilities}
        
        print(f"\n✅ Found {len(working_models)} working models:")
        for model in working_models:
            print(f"   • {model}")
        
        # Run fraud analysis with working models
        print(f"\n📊 RUNNING FRAUD ANALYSIS...")
        
        analysis_results = []
        
        for model_name in working_models:
            print(f"\n🔬 Analyzing with {model_name}...")
            
            # Run different types of analysis based on model capability
            if 'pro' in model_name.lower() or '2.0' in model_name:
                analysis_type = 'detailed'
            elif 'flash' in model_name.lower():
                analysis_type = 'basic'
            else:
                analysis_type = 'basic'
            
            result = await self.analyze_with_model(model_name, analysis_type)
            analysis_results.append(result)
            
            if result['status'] == 'success':
                print(f"   ✅ Success: {result['analysis_length']} chars in {result['response_time']}s")
                print(f"      Preview: {result['analysis'][:100]}...")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
            await asyncio.sleep(5)  # Rate limiting between analyses
        
        # Compile comprehensive results
        comprehensive_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'models_discovered': len(all_models),
            'models_working': len(working_models),
            'models_analyzed': len([r for r in analysis_results if r['status'] == 'success']),
            'fraud_datasets_used': list(self.fraud_samples.keys()),
            'total_fraud_samples': sum(data['total_fraud'] for data in self.fraud_samples.values()),
            'model_capabilities': model_capabilities,
            'analysis_results': analysis_results,
            'model_categories': self.available_models,
            'performance_summary': self.create_performance_summary(analysis_results),
            'recommendations': self.create_recommendations(analysis_results)
        }
        
        # Save results
        output_file = '../data/comprehensive_gemini_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 40)
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Models discovered: {comprehensive_results['models_discovered']}")
        print(f"✅ Models working: {comprehensive_results['models_working']}")
        print(f"🔬 Successful analyses: {comprehensive_results['models_analyzed']}")
        print(f"📈 Fraud samples analyzed: {comprehensive_results['total_fraud_samples']}")
        
        return comprehensive_results
    
    def create_performance_summary(self, analysis_results):
        """Create performance summary of models"""
        successful = [r for r in analysis_results if r['status'] == 'success']
        
        if not successful:
            return {'message': 'No successful analyses'}
        
        return {
            'best_model_by_length': max(successful, key=lambda x: x['analysis_length'])['model'],
            'fastest_model': min(successful, key=lambda x: x['response_time'])['model'],
            'average_response_time': sum(r['response_time'] for r in successful) / len(successful),
            'average_analysis_length': sum(r['analysis_length'] for r in successful) / len(successful),
            'success_rate': len(successful) / len(analysis_results) * 100
        }
    
    def create_recommendations(self, analysis_results):
        """Create recommendations based on results"""
        successful = [r for r in analysis_results if r['status'] == 'success']
        
        if not successful:
            return ['No working models found - check API keys and quotas']
        
        recommendations = []
        
        # Performance-based recommendations
        if len(successful) > 1:
            best_model = max(successful, key=lambda x: x['analysis_length'])['model']
            recommendations.append(f"Use {best_model} for most comprehensive fraud analysis")
            
            fastest_model = min(successful, key=lambda x: x['response_time'])['model']
            recommendations.append(f"Use {fastest_model} for real-time fraud detection")
        
        # Model-specific recommendations
        for result in successful:
            if 'pro' in result['model'].lower():
                recommendations.append(f"{result['model']} suitable for complex fraud pattern analysis")
            elif 'flash' in result['model'].lower():
                recommendations.append(f"{result['model']} suitable for high-volume transaction screening")
        
        return recommendations

def main():
    """Main function"""
    analyzer = ComprehensiveGeminiAnalyzer()
    return asyncio.run(analyzer.run_comprehensive_analysis())

if __name__ == "__main__":
    main()