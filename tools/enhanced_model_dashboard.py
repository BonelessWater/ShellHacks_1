#!/usr/bin/env python3
"""
Enhanced Ultimate Model Comparison Dashboard
Focused on clean UI/UX with no text overlaps and professional presentation
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

class EnhancedModelDashboard:
    """Create enhanced, professional model comparison dashboard"""
    
    def __init__(self):
        """Load all data sources"""
        self.load_all_data()
        
    def load_all_data(self):
        """Load all analysis data from different sources"""
        self.all_results = []
        
        # Load Gemini results
        try:
            with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
                gemini_data = json.load(f)
                gemini_results = gemini_data.get('analysis_results', [])
                
                for result in gemini_results:
                    if result['status'] == 'success':
                        model_name = result['model'].split('/')[-1].replace('gemini-', '').replace('models/', '')
                        # Truncate very long names
                        if len(model_name) > 20:
                            model_name = model_name[:17] + '...'
                        
                        self.all_results.append({
                            'model': model_name,
                            'full_name': result['model'],
                            'provider': 'Gemini',
                            'category': 'Cloud',
                            'analysis_length': result['analysis_length'],
                            'response_time': result['response_time'],
                            'cost_estimate': (result['analysis_length'] / 1000) * 0.002,
                            'efficiency': result['analysis_length'] / result['response_time']
                        })
            print(f"✅ Loaded {len([r for r in self.all_results if r['provider'] == 'Gemini'])} Gemini models")
        except FileNotFoundError:
            print("❌ Gemini data not found")
        
        # Load Ollama results
        try:
            with open('multi_ollama_results.json', 'r') as f:
                ollama_data = json.load(f)
                ollama_results = ollama_data.get('results', [])
                
                for result in ollama_results:
                    self.all_results.append({
                        'model': result['model'],
                        'full_name': result['model'],
                        'provider': 'Ollama',
                        'category': 'Local',
                        'analysis_length': result['analysis_length'],
                        'response_time': result['response_time'],
                        'cost_estimate': 0.0,
                        'efficiency': result.get('efficiency', result['analysis_length'] / result['response_time'])
                    })
            print(f"✅ Loaded {len([r for r in self.all_results if r['provider'] == 'Ollama'])} Ollama models")
        except FileNotFoundError:
            print("❌ Ollama data not found")
    
    def create_clean_dashboard(self):
        """Create clean, professional dashboard with no text overlaps"""
        print("🎨 Creating enhanced UI/UX dashboard...")
        
        if not self.all_results:
            print("❌ No data available")
            return None
        
        # Create dashboard with optimal spacing
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '<b style="font-size:16px; color:#2C3E50;">🏆 Top Performance Models</b>',
                '<b style="font-size:16px; color:#2C3E50;">⚡ Speed vs Quality Matrix</b>',
                '<b style="font-size:16px; color:#2C3E50;">💰 Cost Analysis</b>',
                '<b style="font-size:16px; color:#2C3E50;">🚀 Efficiency Leaderboard</b>',
                '<b style="font-size:16px; color:#2C3E50;">📊 Provider Comparison</b>',
                '<b style="font-size:16px; color:#2C3E50;">🎯 Key Metrics Summary</b>'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "table"}]],
            vertical_spacing=0.18,  # Increased for better separation
            horizontal_spacing=0.15
        )
        
        # 1. Top Performance Models (horizontal bar for better readability)
        top_models = sorted(self.all_results, key=lambda x: x['analysis_length'], reverse=True)[:10]
        
        models = [r['model'] for r in top_models]
        lengths = [r['analysis_length'] for r in top_models]
        providers = [r['provider'] for r in top_models]
        
        colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in providers]
        
        fig.add_trace(go.Bar(
            y=models,
            x=lengths,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'{l:,}' for l in lengths],
            textposition='inside',
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertemplate="<b>%{y}</b><br>Analysis: %{x:,} characters<br>Provider: %{customdata}<br><extra></extra>",
            customdata=providers,
            showlegend=False
        ), row=1, col=1)
        
        # 2. Speed vs Quality Matrix (scatter plot)
        gemini_data = [r for r in self.all_results if r['provider'] == 'Gemini']
        ollama_data = [r for r in self.all_results if r['provider'] == 'Ollama']
        
        # Gemini points
        if gemini_data:
            fig.add_trace(go.Scatter(
                x=[r['response_time'] for r in gemini_data],
                y=[r['analysis_length'] for r in gemini_data],
                mode='markers',
                name='Gemini (Cloud)',
                marker=dict(
                    size=12,
                    color='#4285F4',
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=[r['model'] for r in gemini_data],
                hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<br>Provider: Gemini<extra></extra>"
            ), row=1, col=2)
        
        # Ollama points
        if ollama_data:
            fig.add_trace(go.Scatter(
                x=[r['response_time'] for r in ollama_data],
                y=[r['analysis_length'] for r in ollama_data],
                mode='markers',
                name='Ollama (Local)',
                marker=dict(
                    size=12,
                    color='#FF6B35',
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=[r['model'] for r in ollama_data],
                hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<br>Provider: Ollama<extra></extra>"
            ), row=1, col=2)
        
        # 3. Cost Analysis
        gemini_costs = [r['cost_estimate'] for r in gemini_data] if gemini_data else [0]
        ollama_costs = [0] * len(ollama_data) if ollama_data else [0]
        
        fig.add_trace(go.Bar(
            x=['Gemini (Cloud)', 'Ollama (Local)'],
            y=[sum(gemini_costs)/len(gemini_costs) if gemini_costs else 0, 0],
            marker=dict(
                color=['#4285F4', '#FF6B35'],
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'${sum(gemini_costs)/len(gemini_costs):.4f}' if gemini_costs else '$0', 'FREE'],
            textposition='outside',
            textfont=dict(size=14, color='#2C3E50', family='Arial Black'),
            hovertemplate="<b>%{x}</b><br>Average Cost: %{text}<br><extra></extra>",
            showlegend=False
        ), row=2, col=1)
        
        # 4. Efficiency Leaderboard
        top_efficient = sorted(self.all_results, key=lambda x: x['efficiency'], reverse=True)[:8]
        
        eff_models = [r['model'] for r in top_efficient]
        eff_scores = [r['efficiency'] for r in top_efficient]
        eff_providers = [r['provider'] for r in top_efficient]
        eff_colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in eff_providers]
        
        fig.add_trace(go.Bar(
            x=eff_models,
            y=eff_scores,
            marker=dict(
                color=eff_colors,
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'{s:.0f}' for s in eff_scores],
            textposition='outside',
            textfont=dict(size=12, color='#2C3E50', family='Arial Bold'),
            hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.0f} chars/sec<br>Provider: %{customdata}<br><extra></extra>",
            customdata=eff_providers,
            showlegend=False
        ), row=2, col=2)
        
        # 5. Provider Comparison (Box plot)
        for provider in ['Gemini', 'Ollama']:
            provider_lengths = [r['analysis_length'] for r in self.all_results if r['provider'] == provider]
            if provider_lengths:
                color = '#4285F4' if provider == 'Gemini' else '#FF6B35'
                fig.add_trace(go.Box(
                    y=provider_lengths,
                    name=provider,
                    marker=dict(color=color, opacity=0.8),
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    line=dict(width=2)
                ), row=3, col=1)
        
        # 6. Key Metrics Summary Table
        best_overall = max(self.all_results, key=lambda x: x['analysis_length'])
        best_local = max([r for r in self.all_results if r['provider'] == 'Ollama'], 
                        key=lambda x: x['analysis_length']) if ollama_data else None
        fastest = min(self.all_results, key=lambda x: x['response_time'])
        most_efficient = max(self.all_results, key=lambda x: x['efficiency'])
        
        summary_data = [
            ['🏆 Best Overall', best_overall['model'], f"{best_overall['analysis_length']:,} chars", f"${best_overall['cost_estimate']:.4f}"],
            ['💰 Best Value', best_local['model'] if best_local else 'N/A', 
             f"{best_local['analysis_length']:,} chars" if best_local else 'N/A', 'FREE'],
            ['⚡ Fastest', fastest['model'], f"{fastest['response_time']:.2f} seconds", f"${fastest['cost_estimate']:.4f}"],
            ['🚀 Most Efficient', most_efficient['model'], f"{most_efficient['efficiency']:.0f} chars/sec", f"${most_efficient['cost_estimate']:.4f}"]
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>🎯 Category</b>', '<b>🤖 Model</b>', '<b>📊 Performance</b>', '<b>💰 Cost</b>'],
                fill_color='#2C3E50',
                align='center',
                font=dict(size=14, color='white', family='Arial Black'),
                height=45,
                line=dict(color='white', width=2)
            ),
            cells=dict(
                values=list(zip(*summary_data)),
                fill_color=[['#F8F9FA', '#FFFFFF'] * 2],
                align=['center', 'left', 'center', 'center'],
                font=dict(size=13, color='#2C3E50', family='Arial'),
                height=50,
                line=dict(color='#E8E8E8', width=1)
            )
        ), row=3, col=2)
        
        # Enhanced layout with professional styling
        fig.update_layout(
            title=dict(
                text="<b style='font-size:32px; color:#2C3E50;'>🚀 AI Model Performance Dashboard</b><br>" +
                     f"<span style='color: #7F8C8D; font-size:18px;'>{len([r for r in self.all_results if r['provider'] == 'Gemini'])} Gemini vs " +
                     f"{len([r for r in self.all_results if r['provider'] == 'Ollama'])} Ollama Models • Professional Analysis</span>",
                x=0.5,
                y=0.97,
                font=dict(family='Arial Black')
            ),
            height=1400,  # Optimized height
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="center",
                x=0.5,
                font=dict(size=13, family='Arial Bold'),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#BDC3C7',
                borderwidth=2,
                itemsizing='constant'
            ),
            template="plotly_white",
            font=dict(size=12, family="Arial", color='#2C3E50'),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FAFAFA',
            margin=dict(t=180, b=80, l=120, r=120)
        )
        
        # Clean axis styling with no overlaps
        fig.update_xaxes(
            title_font=dict(size=14, color='#34495E', family='Arial Bold'),
            tickfont=dict(size=11, color='#2C3E50'),
            gridcolor='#E8E8E8',
            gridwidth=1,
            showline=True,
            linecolor='#BDC3C7',
            linewidth=2,
            zeroline=False,
            automargin=True,
            tickangle=0  # Keep most labels horizontal
        )
        
        fig.update_yaxes(
            title_font=dict(size=14, color='#34495E', family='Arial Bold'),
            tickfont=dict(size=11, color='#2C3E50'),
            gridcolor='#E8E8E8',
            gridwidth=1,
            showline=True,
            linecolor='#BDC3C7',
            linewidth=2,
            zeroline=False,
            automargin=True
        )
        
        # Specific axis titles
        fig.update_xaxes(title_text="<b>Analysis Length (Characters)</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>AI Models</b>", row=1, col=1)
        fig.update_xaxes(title_text="<b>Response Time (seconds)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Analysis Length (characters)</b>", row=1, col=2)
        fig.update_xaxes(title_text="<b>Provider Type</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Average Cost per Analysis (USD)</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>AI Models</b>", tickangle=30, row=2, col=2)  # Slight angle for readability
        fig.update_yaxes(title_text="<b>Efficiency (chars/second)</b>", row=2, col=2)
        fig.update_yaxes(title_text="<b>Analysis Length Distribution</b>", row=3, col=1)
        
        # Save dashboard
        dashboard_file = "enhanced_model_dashboard.html"
        fig.write_html(dashboard_file, config={'displayModeBar': True, 'displaylogo': False})
        print(f"✅ Enhanced dashboard created: {dashboard_file}")
        
        return dashboard_file
    
    def open_enhanced_dashboard(self):
        """Create and open the enhanced dashboard"""
        print("🎨 CREATING ENHANCED UI/UX DASHBOARD")
        print("=" * 45)
        
        dashboard_file = self.create_clean_dashboard()
        
        if dashboard_file:
            # Open dashboard
            try:
                abs_path = os.path.abspath(dashboard_file)
                webbrowser.open(f'file://{abs_path}')
                print(f"🚀 Opened: {dashboard_file}")
            except Exception as e:
                print(f"❌ Failed to open: {e}")
            
            # Print summary
            gemini_count = len([r for r in self.all_results if r['provider'] == 'Gemini'])
            ollama_count = len([r for r in self.all_results if r['provider'] == 'Ollama'])
            
            print(f"\n✨ ENHANCED DASHBOARD FEATURES:")
            print("=" * 40)
            print("🎨 Clean, professional design with no text overlaps")
            print("📊 Optimized spacing and improved readability")
            print("🔧 Better axis labels and hover information")
            print("💫 Enhanced color scheme and typography")
            print("📱 Responsive layout with proper margins")
            print()
            print(f"📈 MODELS ANALYZED:")
            print(f"   • {gemini_count} Gemini models (Cloud)")
            print(f"   • {ollama_count} Ollama models (Local)")
            print(f"   • {len(self.all_results)} total models compared")
        
        return dashboard_file

def main():
    """Main function"""
    dashboard = EnhancedModelDashboard()
    return dashboard.open_enhanced_dashboard()

if __name__ == "__main__":
    main()