#!/usr/bin/env python3
"""
Fixed Ultimate Model Comparison Dashboard
Debugging version to fix blank page issue
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

def create_fixed_dashboard():
    """Create a simplified but working dashboard"""
    print("🔧 Creating fixed ultimate model comparison...")
    
    # Load data
    all_results = []
    
    # Load Gemini results
    try:
        with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
            gemini_data = json.load(f)
            gemini_results = gemini_data.get('analysis_results', [])
            
            for result in gemini_results:
                if result['status'] == 'success':
                    model_name = result['model'].split('/')[-1].replace('gemini-', '')
                    # Ensure clean model names
                    if len(model_name) > 25:
                        model_name = model_name[:22] + '...'
                        
                    all_results.append({
                        'model': model_name,
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
    
    print(f"📊 Total models loaded: {len(all_results)}")
    
    # Create simplified dashboard with clear layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '<b style="font-size:18px; color:#1F2937;">Top Performance Models</b>',
            '<b style="font-size:18px; color:#1F2937;">Speed vs Quality</b>',
            '<b style="font-size:18px; color:#1F2937;">Provider Comparison</b>',
            '<b style="font-size:18px; color:#1F2937;">Efficiency Rankings</b>'
        ],
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Top Performance Models (horizontal bars for better readability)
    top_models = sorted(all_results, key=lambda x: x['analysis_length'], reverse=True)[:12]
    
    models = [r['model'] for r in top_models]
    lengths = [r['analysis_length'] for r in top_models]
    providers = [r['provider'] for r in top_models]
    colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in providers]
    
    fig.add_trace(go.Bar(
        y=models,
        x=lengths,
        orientation='h',
        name='Performance',
        marker=dict(color=colors, opacity=0.8),
        text=[f'{l:,}' for l in lengths],
        textposition='inside',
        hovertemplate="<b>%{y}</b><br>Analysis: %{x:,} chars<br>Provider: %{customdata}<extra></extra>",
        customdata=providers,
        showlegend=False
    ), row=1, col=1)
    
    # 2. Speed vs Quality Scatter
    gemini_data = [r for r in all_results if r['provider'] == 'Gemini']
    ollama_data = [r for r in all_results if r['provider'] == 'Ollama']
    
    if gemini_data:
        fig.add_trace(go.Scatter(
            x=[r['response_time'] for r in gemini_data],
            y=[r['analysis_length'] for r in gemini_data],
            mode='markers',
            name='Gemini (Cloud)',
            marker=dict(size=10, color='#4285F4', opacity=0.7),
            text=[r['model'] for r in gemini_data],
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<extra></extra>"
        ), row=1, col=2)
    
    if ollama_data:
        fig.add_trace(go.Scatter(
            x=[r['response_time'] for r in ollama_data],
            y=[r['analysis_length'] for r in ollama_data],
            mode='markers',
            name='Ollama (Local)',
            marker=dict(size=10, color='#FF6B35', opacity=0.7),
            text=[r['model'] for r in ollama_data],
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<extra></extra>"
        ), row=1, col=2)
    
    # 3. Provider Comparison
    providers = ['Gemini (Cloud)', 'Ollama (Local)']
    model_counts = [len(gemini_data), len(ollama_data)]
    
    fig.add_trace(go.Bar(
        x=providers,
        y=model_counts,
        name='Model Count',
        marker=dict(color=['#4285F4', '#FF6B35'], opacity=0.8),
        text=model_counts,
        textposition='outside',
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
        name='Efficiency',
        marker=dict(color=eff_colors, opacity=0.8),
        text=[f'{s:.0f}' for s in eff_scores],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.0f} chars/sec<extra></extra>",
        showlegend=False
    ), row=2, col=2)
    
    # Clean, professional layout
    fig.update_layout(
        title=dict(
            text="<b style='font-size:28px; color:#1F2937;'>AI Model Performance Dashboard</b><br>" +
                 f"<span style='color:#6B7280; font-size:16px;'>{len(all_results)} Models Analyzed • Professional Analysis</span>",
            x=0.5,
            y=0.95
        ),
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template="plotly_white",
        font=dict(size=12, color='#1F2937'),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F9FAFB',
        margin=dict(t=120, b=60, l=80, r=80)
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='#E5E7EB',
        showline=True,
        linecolor='#D1D5DB',
        automargin=True
    )
    
    fig.update_yaxes(
        gridcolor='#E5E7EB',
        showline=True,
        linecolor='#D1D5DB',
        automargin=True
    )
    
    # Add axis labels
    fig.update_xaxes(title_text="<b>Analysis Length (Characters)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>AI Models</b>", row=1, col=1)
    fig.update_xaxes(title_text="<b>Response Time (seconds)</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Analysis Length</b>", row=1, col=2)
    fig.update_xaxes(title_text="<b>Provider</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Model Count</b>", row=2, col=1)
    fig.update_xaxes(title_text="<b>Models</b>", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="<b>Efficiency (chars/sec)</b>", row=2, col=2)
    
    # Save the dashboard
    dashboard_file = "fixed_ultimate_comparison.html"
    
    try:
        # Use write_html with explicit configuration
        fig.write_html(
            dashboard_file,
            include_plotlyjs=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'model_comparison',
                    'height': 900,
                    'width': 1200,
                    'scale': 2
                }
            },
            div_id="model-comparison-dashboard"
        )
        
        print(f"✅ Fixed dashboard created: {dashboard_file}")
        
        # Open the dashboard
        abs_path = os.path.abspath(dashboard_file)
        webbrowser.open(f'file://{abs_path}')
        print(f"🚀 Opened: {dashboard_file}")
        
        # Print summary
        print(f"\n📊 DASHBOARD SUMMARY:")
        print("=" * 30)
        print(f"• {len(gemini_data)} Gemini models")
        print(f"• {len(ollama_data)} Ollama models")
        print(f"• {len(all_results)} total models")
        
        if all_results:
            best = max(all_results, key=lambda x: x['analysis_length'])
            fastest = min(all_results, key=lambda x: x['response_time'])
            print(f"• Best: {best['model']} ({best['analysis_length']:,} chars)")
            print(f"• Fastest: {fastest['model']} ({fastest['response_time']:.2f}s)")
        
        return dashboard_file
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    create_fixed_dashboard()