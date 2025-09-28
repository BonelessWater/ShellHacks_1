#!/usr/bin/env python3
"""
Working Ultimate Model Comparison Dashboard
Fixed version that guarantees proper HTML rendering
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os

def create_working_dashboard():
    """Create a guaranteed working dashboard"""
    print("🔧 Creating working ultimate model comparison dashboard...")
    
    # Load data with error handling
    all_results = []
    
    # Load Gemini results
    try:
        with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
            gemini_data = json.load(f)
            gemini_results = gemini_data.get('analysis_results', [])
            
            for result in gemini_results:
                if result['status'] == 'success':
                    model_name = result['model'].split('/')[-1].replace('gemini-', '').replace('models/', '')
                    # Clean model names for display
                    if len(model_name) > 20:
                        model_name = model_name[:17] + '...'
                        
                    all_results.append({
                        'model': model_name,
                        'full_name': result['model'],
                        'provider': 'Gemini',
                        'analysis_length': result['analysis_length'],
                        'response_time': result['response_time'],
                        'cost_estimate': (result['analysis_length'] / 1000) * 0.002,
                        'efficiency': result['analysis_length'] / result['response_time']
                    })
        print(f"✅ Loaded {len([r for r in all_results if r['provider'] == 'Gemini'])} Gemini models")
    except Exception as e:
        print(f"❌ Error loading Gemini data: {e}")
    
    # Load Ollama results
    try:
        with open('multi_ollama_results.json', 'r') as f:
            ollama_data = json.load(f)
            ollama_results = ollama_data.get('results', [])
            
            for result in ollama_results:
                all_results.append({
                    'model': result['model'],
                    'full_name': result['model'],
                    'provider': 'Ollama',
                    'analysis_length': result['analysis_length'],
                    'response_time': result['response_time'],
                    'cost_estimate': 0.0,
                    'efficiency': result.get('efficiency', result['analysis_length'] / result['response_time'])
                })
        print(f"✅ Loaded {len([r for r in all_results if r['provider'] == 'Ollama'])} Ollama models")
    except Exception as e:
        print(f"❌ Error loading Ollama data: {e}")
    
    if not all_results:
        print("❌ No data loaded! Cannot create dashboard.")
        return None
    
    print(f"📊 Creating dashboard with {len(all_results)} models")
    
    # Create figure with proper subplot configuration
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Top Performance Models',
            'Speed vs Quality Analysis', 
            'Provider Comparison',
            'Efficiency Rankings'
        ],
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Top Performance - Horizontal bars for better readability
    top_models = sorted(all_results, key=lambda x: x['analysis_length'], reverse=True)[:12]
    
    models = [r['model'] for r in top_models]
    lengths = [r['analysis_length'] for r in top_models]
    providers = [r['provider'] for r in top_models]
    colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in providers]
    
    fig.add_trace(go.Bar(
        y=models,
        x=lengths,
        orientation='h',
        marker=dict(color=colors, opacity=0.8, line=dict(color='white', width=1)),
        text=[f'{l:,}' for l in lengths],
        textposition='inside',
        textfont=dict(color='white', size=10),
        hovertemplate="<b>%{y}</b><br>Analysis: %{x:,} chars<br>Provider: %{customdata}<extra></extra>",
        customdata=providers,
        showlegend=False
    ), row=1, col=1)
    
    # 2. Speed vs Quality Scatter Plot
    gemini_models = [r for r in all_results if r['provider'] == 'Gemini']
    ollama_models = [r for r in all_results if r['provider'] == 'Ollama']
    
    if gemini_models:
        fig.add_trace(go.Scatter(
            x=[r['response_time'] for r in gemini_models],
            y=[r['analysis_length'] for r in gemini_models],
            mode='markers',
            name='Gemini (Cloud)',
            marker=dict(size=10, color='#4285F4', opacity=0.7, line=dict(color='white', width=1)),
            text=[r['model'] for r in gemini_models],
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<extra></extra>"
        ), row=1, col=2)
    
    if ollama_models:
        fig.add_trace(go.Scatter(
            x=[r['response_time'] for r in ollama_models],
            y=[r['analysis_length'] for r in ollama_models],
            mode='markers',
            name='Ollama (Local)',
            marker=dict(size=10, color='#FF6B35', opacity=0.7, line=dict(color='white', width=1)),
            text=[r['model'] for r in ollama_models],
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<extra></extra>"
        ), row=1, col=2)
    
    # 3. Provider Comparison
    provider_counts = [len(gemini_models), len(ollama_models)]
    provider_names = ['Gemini (Cloud)', 'Ollama (Local)']
    
    fig.add_trace(go.Bar(
        x=provider_names,
        y=provider_counts,
        marker=dict(color=['#4285F4', '#FF6B35'], opacity=0.8, line=dict(color='white', width=1)),
        text=provider_counts,
        textposition='outside',
        textfont=dict(size=12, color='#333'),
        hovertemplate="<b>%{x}</b><br>Models: %{y}<extra></extra>",
        showlegend=False
    ), row=2, col=1)
    
    # 4. Efficiency Rankings
    top_efficient = sorted(all_results, key=lambda x: x['efficiency'], reverse=True)[:8]
    
    eff_models = [r['model'] for r in top_efficient]
    eff_scores = [r['efficiency'] for r in top_efficient]
    eff_providers = [r['provider'] for r in top_efficient]
    eff_colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in eff_providers]
    
    fig.add_trace(go.Bar(
        x=eff_models,
        y=eff_scores,
        marker=dict(color=eff_colors, opacity=0.8, line=dict(color='white', width=1)),
        text=[f'{s:.0f}' for s in eff_scores],
        textposition='outside',
        textfont=dict(size=10, color='#333'),
        hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.0f} chars/sec<extra></extra>",
        showlegend=False
    ), row=2, col=2)
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text="<b>AI Model Performance Dashboard</b><br>" +
                 f"<span style='font-size:14px; color:#666;'>{len(all_results)} Models Analyzed • Professional Analysis</span>",
            x=0.5,
            font=dict(size=24, color='#333')
        ),
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        template="plotly_white",
        font=dict(size=11, color='#333'),
        margin=dict(t=120, b=60, l=80, r=80)
    )
    
    # Update axes with clear labels
    fig.update_xaxes(
        gridcolor='#E5E7EB',
        showline=True,
        linecolor='#D1D5DB',
        linewidth=1,
        automargin=True
    )
    
    fig.update_yaxes(
        gridcolor='#E5E7EB', 
        showline=True,
        linecolor='#D1D5DB',
        linewidth=1,
        automargin=True
    )
    
    # Add axis titles
    fig.update_xaxes(title_text="Analysis Length (Characters)", row=1, col=1)
    fig.update_yaxes(title_text="AI Models", row=1, col=1)
    fig.update_xaxes(title_text="Response Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Analysis Length", row=1, col=2)
    fig.update_xaxes(title_text="Provider", row=2, col=1)
    fig.update_yaxes(title_text="Model Count", row=2, col=1)
    fig.update_xaxes(title_text="Models", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (chars/sec)", row=2, col=2)
    
    # Save with robust configuration
    output_file = "ultimate_model_comparison.html"
    
    try:
        # Create the HTML with full Plotly.js inclusion
        html_content = fig.to_html(
            include_plotlyjs=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'model_comparison',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                },
                'responsive': True
            },
            div_id="model-dashboard"
        )
        
        # Add some custom styling for better appearance
        enhanced_html = html_content.replace(
            '<head>',
            '''<head>
            <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px;
            }
            .plotly-graph-div {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 10px;
            }
            </style>'''
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_html)
        
        print(f"✅ Dashboard created successfully: {output_file}")
        
        # Verify file was created properly
        file_size = os.path.getsize(output_file)
        print(f"📁 File size: {file_size:,} bytes")
        
        if file_size < 100000:  # Less than 100KB indicates a problem
            print("⚠️  File seems small, there might be an issue")
            return None
        
        # Open in browser
        abs_path = os.path.abspath(output_file)
        webbrowser.open(f'file://{abs_path}')
        print(f"🚀 Opened dashboard in browser")
        
        # Print summary
        print(f"\n📊 DASHBOARD SUMMARY:")
        print("=" * 30)
        print(f"• Total models: {len(all_results)}")
        print(f"• Gemini models: {len(gemini_models)}")
        print(f"• Ollama models: {len(ollama_models)}")
        
        if all_results:
            best = max(all_results, key=lambda x: x['analysis_length'])
            fastest = min(all_results, key=lambda x: x['response_time'])
            most_efficient = max(all_results, key=lambda x: x['efficiency'])
            
            print(f"• Best overall: {best['model']} ({best['analysis_length']:,} chars)")
            print(f"• Fastest: {fastest['model']} ({fastest['response_time']:.2f}s)")
            print(f"• Most efficient: {most_efficient['model']} ({most_efficient['efficiency']:.0f} c/s)")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    create_working_dashboard()