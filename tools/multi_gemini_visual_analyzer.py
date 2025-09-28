#!/usr/bin/env python3
"""
Multi-Gemini Visual Comparison Dashboard
Creates interactive visualizations comparing different Gemini models on fraud detection
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

def create_multi_gemini_dashboard():
    """Create comprehensive multi-Gemini model comparison dashboard"""
    
    print("🎨 Creating Multi-Gemini Model Comparison Dashboard...")
    print("=" * 55)
    
    # Load comparison results
    try:
        with open('../data/multi_gemini_fraud_comparison.json', 'r') as f:
            comparison_data = json.load(f)
    except FileNotFoundError:
        print("❌ Multi-Gemini comparison file not found. Run multi_gemini_model_analyzer.py first.")
        return None
    
    # Extract data
    model_results = comparison_data['model_results']
    performance_analysis = comparison_data['performance_analysis']
    model_configs = comparison_data['model_configurations']
    
    print(f"📊 Processing {len(model_results)} model test results...")
    
    # Create comprehensive dashboard
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            '🏆 Model Success Rates', '📝 Analysis Length by Model',
            '⚡ Response Time Comparison', '🎯 Prompt Success by Model',
            '📊 Model Capability Matrix', '🔄 Success Rate by Prompt Type',
            '💡 Quality vs Speed Analysis', '🥇 Overall Model Ranking'
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Extract successful results for analysis
    successful_results = [r for r in model_results if 'error' not in r]
    failed_results = [r for r in model_results if 'error' in r]
    
    # 1. Model Success Rates
    model_success = {}
    for model_name, perf in performance_analysis['model_performance'].items():
        model_success[model_name] = perf['success_rate']
    
    if model_success:
        models = list(model_success.keys())
        success_rates = list(model_success.values())
        colors = ['#28A745' if x == 100 else '#FFC107' if x >= 66 else '#DC3545' for x in success_rates]
        
        fig.add_trace(go.Bar(
            x=models, y=success_rates, name="Success Rate %",
            marker_color=colors,
            text=[f'{x:.1f}%' for x in success_rates],
            textposition='auto'
        ), row=1, col=1)
    
    # 2. Analysis Length by Model
    if successful_results:
        model_lengths = {}
        for result in successful_results:
            model_name = result['model_name']
            if model_name not in model_lengths:
                model_lengths[model_name] = []
            model_lengths[model_name].append(result.get('analysis_length', 0))
        
        models = list(model_lengths.keys())
        avg_lengths = [sum(lengths)/len(lengths) if lengths else 0 for lengths in model_lengths.values()]
        
        fig.add_trace(go.Bar(
            x=models, y=avg_lengths, name="Avg Analysis Length",
            marker_color='#4ECDC4',
            text=[f'{x:.0f}' for x in avg_lengths],
            textposition='auto'
        ), row=1, col=2)
    
    # 3. Response Time Comparison
    if successful_results:
        model_times = {}
        for result in successful_results:
            model_name = result['model_name']
            if model_name not in model_times:
                model_times[model_name] = []
            model_times[model_name].append(result.get('response_time', 0))
        
        models = list(model_times.keys())
        avg_times = [sum(times)/len(times) if times else 0 for times in model_times.values()]
        
        fig.add_trace(go.Bar(
            x=models, y=avg_times, name="Avg Response Time (s)",
            marker_color='#45B7D1',
            text=[f'{x:.2f}s' for x in avg_times],
            textposition='auto'
        ), row=2, col=1)
    
    # 4. Prompt Success by Model
    prompt_success = {}
    for result in successful_results:
        model_name = result['model_name']
        if model_name not in prompt_success:
            prompt_success[model_name] = 0
        prompt_success[model_name] += 1
    
    total_prompts = comparison_data['total_prompts_tested']
    if prompt_success:
        models = list(prompt_success.keys())
        success_counts = [prompt_success.get(model, 0) for model in models]
        
        fig.add_trace(go.Bar(
            x=models, y=success_counts, name="Successful Prompts",
            marker_color='#96CEB4',
            text=[f'{x}/{total_prompts}' for x in success_counts],
            textposition='auto'
        ), row=2, col=2)
    
    # 5. Model Capability Matrix (Heatmap-style bar chart)
    if successful_results:
        # Create capability matrix
        models = list(set(r['model_name'] for r in successful_results))
        prompts = list(set(r['prompt_type'] for r in successful_results))
        
        capability_data = []
        for model in models:
            model_capabilities = []
            for prompt in prompts:
                # Check if model succeeded on this prompt
                success = any(r['model_name'] == model and r['prompt_type'] == prompt 
                            for r in successful_results)
                model_capabilities.append(100 if success else 0)
            capability_data.extend(model_capabilities)
        
        # Create stacked bars for each prompt type
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, prompt in enumerate(prompts):
            prompt_data = []
            for j, model in enumerate(models):
                success = any(r['model_name'] == model and r['prompt_type'] == prompt 
                            for r in successful_results)
                prompt_data.append(100 if success else 0)
            
            fig.add_trace(go.Bar(
                name=prompt.replace('_', ' ').title(),
                x=models, y=prompt_data,
                marker_color=colors[i % len(colors)],
                opacity=0.8
            ), row=3, col=1)
    
    # 6. Success Rate by Prompt Type
    prompt_perf = performance_analysis.get('prompt_performance', {})
    if prompt_perf:
        prompts = list(prompt_perf.keys())
        success_counts = [prompt_perf[p]['successful_models'] for p in prompts]
        total_models = comparison_data['total_models_tested']
        
        fig.add_trace(go.Bar(
            x=prompts, y=success_counts, name="Models Succeeded",
            marker_color='#FFBE0B',
            text=[f'{x}/{total_models}' for x in success_counts],
            textposition='auto'
        ), row=3, col=2)
    
    # 7. Quality vs Speed Analysis
    if successful_results:
        model_stats = {}
        for result in successful_results:
            model_name = result['model_name']
            if model_name not in model_stats:
                model_stats[model_name] = {'lengths': [], 'times': []}
            model_stats[model_name]['lengths'].append(result.get('analysis_length', 0))
            model_stats[model_name]['times'].append(result.get('response_time', 0))
        
        for model_name, stats in model_stats.items():
            avg_length = sum(stats['lengths']) / len(stats['lengths']) if stats['lengths'] else 0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            
            fig.add_trace(go.Scatter(
                x=[avg_time], y=[avg_length],
                mode='markers+text',
                text=[model_name.split()[-1]],  # Just model version
                textposition="top center",
                marker=dict(size=15, opacity=0.8),
                name=model_name
            ), row=4, col=1)
    
    # 8. Overall Model Ranking
    if performance_analysis['model_performance']:
        model_perf = performance_analysis['model_performance']
        sorted_models = sorted(model_perf.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        models = [item[0] for item in sorted_models]
        scores = [item[1]['success_rate'] for item in sorted_models]
        colors = ['#28A745', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFBE0B'][:len(models)]
        
        fig.add_trace(go.Bar(
            x=models, y=scores, name="Overall Score",
            marker_color=colors,
            text=[f'{x:.1f}%' for x in scores],
            textposition='auto'
        ), row=4, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="🎯 Multi-Gemini Model Fraud Detection Comparison<br><sub>Performance Analysis Across Different Gemini Variants | ShellHacks 2025</sub>",
        title_x=0.5,
        height=1600,
        showlegend=True,
        template="plotly_white",
        font=dict(size=10)
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Success Rate %", row=1, col=1)
    fig.update_yaxes(title_text="Characters", row=1, col=2)
    fig.update_yaxes(title_text="Seconds", row=2, col=1)
    fig.update_yaxes(title_text="Successful Prompts", row=2, col=2)
    fig.update_yaxes(title_text="Success %", row=3, col=1)
    fig.update_yaxes(title_text="Models Succeeded", row=3, col=2)
    fig.update_yaxes(title_text="Analysis Length", row=4, col=1)
    fig.update_xaxes(title_text="Response Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Overall Score", row=4, col=2)
    
    # Save dashboard
    dashboard_file = "multi_gemini_model_dashboard.html"
    fig.write_html(dashboard_file)
    print(f"✅ Multi-Gemini dashboard created: {dashboard_file}")
    
    # Create model comparison table
    model_table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=['🤖 Gemini Model', '📊 Success Rate', '📝 Avg Analysis Length', '⚡ Avg Response Time', '🎯 Prompts Completed', '💡 Best For'],
            fill_color='#4285F4',
            align='center',
            font=dict(size=13, color='white'),
            height=40
        ),
        cells=dict(
            values=[
                [info['name'] for info in model_configs.values()],
                [f"{performance_analysis['model_performance'].get(info['name'], {}).get('success_rate', 0):.1f}%" 
                 for info in model_configs.values()],
                [f"{performance_analysis['model_performance'].get(info['name'], {}).get('avg_analysis_length', 0):.0f} chars" 
                 for info in model_configs.values()],
                [f"{performance_analysis['model_performance'].get(info['name'], {}).get('avg_response_time', 0):.2f}s" 
                 for info in model_configs.values()],
                [f"{performance_analysis['model_performance'].get(info['name'], {}).get('successful_tests', 0)}/{comparison_data['total_prompts_tested']}" 
                 for info in model_configs.values()],
                [info['description'] for info in model_configs.values()]
            ],
            fill_color=[['#f8f9fa', '#e9ecef'] * 3],
            align='left',
            font=dict(size=11),
            height=35
        )
    )])
    
    model_table_fig.update_layout(
        title="📋 Detailed Gemini Model Comparison Table",
        height=400,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    
    table_file = "gemini_model_comparison_table.html"
    model_table_fig.write_html(table_file)
    print(f"✅ Model comparison table created: {table_file}")
    
    # Open visualizations
    print(f"\n🌐 Opening Gemini model comparison visualizations...")
    
    files = [dashboard_file, table_file]
    for file in files:
        abs_path = os.path.abspath(file)
        webbrowser.open(f'file://{abs_path}')
        print(f"🚀 Opened: {file}")
    
    # Print summary
    print(f"\n🎉 MULTI-GEMINI VISUAL ANALYSIS COMPLETE!")
    print("=" * 45)
    print("📊 Created Files:")
    print(f"  1. {dashboard_file} - Complete model comparison dashboard")
    print(f"  2. {table_file} - Detailed model performance table")
    
    if performance_analysis.get('best_model'):
        print(f"\n🏆 Best Performing Model: {performance_analysis['best_model']}")
    
    print(f"\n📈 Analysis Summary:")
    print(f"   • Models Tested: {comparison_data['total_models_tested']}")
    print(f"   • Prompt Types: {comparison_data['total_prompts_tested']}")
    print(f"   • Success Rate: {performance_analysis['success_rate']:.1f}%")
    print(f"   • Total Tests: {comparison_data['total_combinations']}")
    
    return files

def main():
    """Main function"""
    return create_multi_gemini_dashboard()

if __name__ == "__main__":
    main()