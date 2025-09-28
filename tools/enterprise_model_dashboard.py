#!/usr/bin/env python3
"""
Ultimate Professional Dashboard Suite
Comprehensive, clean, and enterprise-ready visualization system
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

class UltimateProfessionalDashboard:
    """Create the ultimate professional dashboard with modern UI/UX"""
    
    def __init__(self):
        """Initialize with enterprise design system"""
        # Professional enterprise color palette
        self.colors = {
            'primary': '#0F172A',      # Slate 900
            'secondary': '#1E293B',    # Slate 800
            'accent': '#3B82F6',       # Blue 500
            'success': '#10B981',      # Emerald 500
            'warning': '#F59E0B',      # Amber 500
            'error': '#EF4444',        # Red 500
            'gemini': '#4285F4',       # Google Blue
            'ollama': '#F97316',       # Orange 500
            'background': '#FFFFFF',   # White
            'surface': '#F8FAFC',      # Slate 50
            'border': '#E2E8F0',       # Slate 200
            'text_primary': '#1E293B', # Slate 800
            'text_secondary': '#64748B', # Slate 500
            'text_muted': '#94A3B8'    # Slate 400
        }
        
        # Enterprise typography system
        self.fonts = {
            'primary': 'Inter, system-ui, -apple-system, sans-serif',
            'mono': 'JetBrains Mono, Consolas, monospace',
            'display': 'Inter, system-ui, sans-serif'
        }
        
        self.load_comprehensive_data()
        
    def load_comprehensive_data(self):
        """Load all available model data"""
        self.all_results = []
        
        # Load Gemini results
        try:
            with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
                gemini_data = json.load(f)
                gemini_results = gemini_data.get('analysis_results', [])
                
                for result in gemini_results:
                    if result['status'] == 'success':
                        model_name = result['model'].split('/')[-1]
                        model_name = model_name.replace('gemini-', '').replace('models/', '')
                        
                        # Clean up very long names
                        display_name = model_name[:20] + '...' if len(model_name) > 20 else model_name
                        
                        self.all_results.append({
                            'model': display_name,
                            'full_name': result['model'],
                            'provider': 'Gemini',
                            'category': 'Cloud',
                            'analysis_length': result['analysis_length'],
                            'response_time': result['response_time'],
                            'cost_estimate': (result['analysis_length'] / 1000) * 0.002,
                            'efficiency': result['analysis_length'] / result['response_time'],
                            'quality_score': min(result['analysis_length'] / 100, 100),
                            'color': self.colors['gemini']
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
                        'efficiency': result.get('efficiency', result['analysis_length'] / result['response_time']),
                        'quality_score': min(result['analysis_length'] / 100, 100),
                        'color': self.colors['ollama']
                    })
            print(f"✅ Loaded {len([r for r in self.all_results if r['provider'] == 'Ollama'])} Ollama models")
        except FileNotFoundError:
            print("❌ Ollama data not found")
    
    def create_enterprise_dashboard(self):
        """Create enterprise-grade dashboard with perfect UI/UX"""
        print("🏢 Creating enterprise-grade dashboard...")
        
        if not self.all_results:
            print("❌ No data available")
            return None
        
        # Create enterprise layout with perfect spacing
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '<b style="font-size:16px; color:#1E293B; font-weight:600;">Performance Leaderboard</b>',
                '<b style="font-size:16px; color:#1E293B; font-weight:600;">Cost vs Quality Matrix</b>',
                '<b style="font-size:16px; color:#1E293B; font-weight:600;">Provider Analysis</b>',
                '<b style="font-size:16px; color:#1E293B; font-weight:600;">Efficiency Rankings</b>',
                '<b style="font-size:16px; color:#1E293B; font-weight:600;">Model Distribution</b>',
                '<b style="font-size:16px; color:#1E293B; font-weight:600;">Executive Summary</b>'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # 1. Performance Leaderboard - Clean horizontal bars
        top_performers = sorted(self.all_results, key=lambda x: x['analysis_length'], reverse=True)[:12]
        
        models = [r['model'] for r in top_performers]
        lengths = [r['analysis_length'] for r in top_performers]
        providers = [r['provider'] for r in top_performers]
        colors = [r['color'] for r in top_performers]
        
        fig.add_trace(go.Bar(
            y=models,
            x=lengths,
            orientation='h',
            marker=dict(
                color=colors,
                opacity=0.9,
                line=dict(color='white', width=1.5)
            ),
            text=[f'{l:,}' for l in lengths],
            textposition='inside',
            textfont=dict(
                size=11, 
                color='white', 
                family=self.fonts['primary']
            ),
            hovertemplate="<b style='font-family:Inter; font-weight:600;'>%{y}</b><br>" +
                         "Analysis: <b>%{x:,}</b> characters<br>" +
                         "Provider: <b>%{customdata}</b><br>" +
                         "<extra></extra>",
            customdata=providers,
            showlegend=False
        ), row=1, col=1)
        
        # 2. Cost vs Quality Matrix - Professional scatter
        gemini_data = [r for r in self.all_results if r['provider'] == 'Gemini']
        ollama_data = [r for r in self.all_results if r['provider'] == 'Ollama']
        
        if gemini_data:
            fig.add_trace(go.Scatter(
                x=[r['cost_estimate'] for r in gemini_data],
                y=[r['quality_score'] for r in gemini_data],
                mode='markers',
                name='Cloud Models',
                marker=dict(
                    size=14,
                    color=self.colors['gemini'],
                    opacity=0.8,
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                text=[r['model'] for r in gemini_data],
                hovertemplate="<b style='font-family:Inter; font-weight:600;'>%{text}</b><br>" +
                             "Cost: <b>$%{x:.4f}</b><br>" +
                             "Quality: <b>%{y:.1f}/100</b><br>" +
                             "Type: <b>Cloud</b><extra></extra>"
            ), row=1, col=2)
        
        if ollama_data:
            fig.add_trace(go.Scatter(
                x=[0.0001 for _ in ollama_data],  # Small offset for visibility
                y=[r['quality_score'] for r in ollama_data],
                mode='markers',
                name='Local Models',
                marker=dict(
                    size=14,
                    color=self.colors['ollama'],
                    opacity=0.8,
                    line=dict(color='white', width=2),
                    symbol='diamond'
                ),
                text=[r['model'] for r in ollama_data],
                hovertemplate="<b style='font-family:Inter; font-weight:600;'>%{text}</b><br>" +
                             "Cost: <b>FREE</b><br>" +
                             "Quality: <b>%{y:.1f}/100</b><br>" +
                             "Type: <b>Local</b><extra></extra>"
            ), row=1, col=2)
        
        # 3. Provider Analysis - Professional comparison
        providers = ['Gemini (Cloud)', 'Ollama (Local)']
        model_counts = [
            len([r for r in self.all_results if r['provider'] == 'Gemini']),
            len([r for r in self.all_results if r['provider'] == 'Ollama'])
        ]
        avg_qualities = []
        
        for provider in ['Gemini', 'Ollama']:
            provider_models = [r for r in self.all_results if r['provider'] == provider]
            if provider_models:
                avg_quality = sum(r['quality_score'] for r in provider_models) / len(provider_models)
                avg_qualities.append(avg_quality)
            else:
                avg_qualities.append(0)
        
        fig.add_trace(go.Bar(
            x=providers,
            y=avg_qualities,
            marker=dict(
                color=[self.colors['gemini'], self.colors['ollama']],
                opacity=0.85,
                line=dict(color='white', width=2)
            ),
            text=[f'{q:.1f}' for q in avg_qualities],
            textposition='outside',
            textfont=dict(
                size=14, 
                color=self.colors['text_primary'], 
                family=self.fonts['primary']
            ),
            hovertemplate="<b style='font-family:Inter; font-weight:600;'>%{x}</b><br>" +
                         "Avg Quality: <b>%{y:.1f}/100</b><br>" +
                         "Models: <b>%{customdata}</b><br>" +
                         "<extra></extra>",
            customdata=model_counts,
            showlegend=False
        ), row=2, col=1)
        
        # 4. Efficiency Rankings - Top performers
        top_efficient = sorted(self.all_results, key=lambda x: x['efficiency'], reverse=True)[:10]
        
        eff_models = [r['model'] for r in top_efficient]
        eff_scores = [r['efficiency'] for r in top_efficient]
        eff_colors = [r['color'] for r in top_efficient]
        
        fig.add_trace(go.Bar(
            x=eff_models,
            y=eff_scores,
            marker=dict(
                color=eff_colors,
                opacity=0.85,
                line=dict(color='white', width=2)
            ),
            text=[f'{s:.0f}' for s in eff_scores],
            textposition='outside',
            textfont=dict(
                size=11, 
                color=self.colors['text_primary'], 
                family=self.fonts['primary']
            ),
            hovertemplate="<b style='font-family:Inter; font-weight:600;'>%{x}</b><br>" +
                         "Efficiency: <b>%{y:.0f}</b> chars/sec<br>" +
                         "<extra></extra>",
            showlegend=False
        ), row=2, col=2)
        
        # 5. Model Distribution - Professional pie chart
        provider_counts = {}
        for result in self.all_results:
            provider = result['provider']
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        fig.add_trace(go.Pie(
            labels=list(provider_counts.keys()),
            values=list(provider_counts.values()),
            marker=dict(
                colors=[self.colors['gemini'], self.colors['ollama']],
                line=dict(color='white', width=3)
            ),
            textfont=dict(
                size=13, 
                color='white', 
                family=self.fonts['primary']
            ),
            hovertemplate="<b style='font-family:Inter; font-weight:600;'>%{label}</b><br>" +
                         "Models: <b>%{value}</b><br>" +
                         "Percentage: <b>%{percent}</b><br>" +
                         "<extra></extra>",
            pull=[0.1, 0.1],
            hole=0.4
        ), row=3, col=1)
        
        # 6. Executive Summary Table
        best_overall = max(self.all_results, key=lambda x: x['analysis_length'])
        best_local = max([r for r in self.all_results if r['provider'] == 'Ollama'], 
                        key=lambda x: x['analysis_length']) if ollama_data else None
        fastest = min(self.all_results, key=lambda x: x['response_time'])
        most_efficient = max(self.all_results, key=lambda x: x['efficiency'])
        
        summary_data = [
            ['🏆 Best Overall', best_overall['model'], f"{best_overall['analysis_length']:,} chars"],
            ['💰 Best Value', best_local['model'] if best_local else 'N/A', 'FREE'],
            ['⚡ Fastest', fastest['model'], f"{fastest['response_time']:.2f}s"],
            ['🚀 Most Efficient', most_efficient['model'], f"{most_efficient['efficiency']:.0f} c/s"],
            ['📊 Total Models', f"{len(self.all_results)}", f"{len(gemini_data)}+{len(ollama_data)}"]
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Winner</b>', '<b>Value</b>'],
                fill_color=self.colors['primary'],
                align='center',
                font=dict(
                    size=13, 
                    color='white', 
                    family=self.fonts['primary']
                ),
                height=45,
                line=dict(color='white', width=2)
            ),
            cells=dict(
                values=list(zip(*summary_data)),
                fill_color=[['#F8FAFC', '#FFFFFF'] * 3],
                align=['left', 'left', 'center'],
                font=dict(
                    size=12, 
                    color=self.colors['text_primary'], 
                    family=self.fonts['primary']
                ),
                height=42,
                line=dict(color=self.colors['border'], width=1)
            )
        ), row=3, col=2)
        
        # Enterprise layout with perfect typography
        fig.update_layout(
            title=dict(
                text="<b style='font-size:32px; color:#0F172A; font-family:Inter; font-weight:700;'>" +
                     "AI Model Performance Dashboard</b><br>" +
                     f"<span style='color:#64748B; font-size:16px; font-family:Inter; font-weight:400;'>" +
                     f"Enterprise Analysis • {len(self.all_results)} Models • Professional Grade</span>",
                x=0.5,
                y=0.97,
                xanchor='center'
            ),
            height=1100,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(
                    size=14, 
                    family=self.fonts['primary'], 
                    color=self.colors['text_primary']
                ),
                bgcolor='rgba(255,255,255,0.98)',
                bordercolor=self.colors['border'],
                borderwidth=1,
                itemsizing='constant'
            ),
            template="plotly_white",
            font=dict(
                size=12, 
                family=self.fonts['primary'], 
                color=self.colors['text_primary']
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            margin=dict(t=160, b=80, l=100, r=100)
        )
        
        # Perfect axis styling
        fig.update_xaxes(
            title_font=dict(
                size=13, 
                color=self.colors['text_primary'], 
                family=self.fonts['primary']
            ),
            tickfont=dict(
                size=11, 
                color=self.colors['text_secondary'], 
                family=self.fonts['primary']
            ),
            gridcolor='#F1F5F9',
            gridwidth=1,
            showline=True,
            linecolor=self.colors['border'],
            linewidth=1.5,
            zeroline=False,
            automargin=True
        )
        
        fig.update_yaxes(
            title_font=dict(
                size=13, 
                color=self.colors['text_primary'], 
                family=self.fonts['primary']
            ),
            tickfont=dict(
                size=11, 
                color=self.colors['text_secondary'], 
                family=self.fonts['primary']
            ),
            gridcolor='#F1F5F9',
            gridwidth=1,
            showline=True,
            linecolor=self.colors['border'],
            linewidth=1.5,
            zeroline=False,
            automargin=True
        )
        
        # Professional axis labels
        fig.update_xaxes(title_text="<b>Analysis Output (Characters)</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>AI Models</b>", row=1, col=1)
        fig.update_xaxes(title_text="<b>Cost per Analysis (USD)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Quality Score (0-100)</b>", row=1, col=2)
        fig.update_xaxes(title_text="<b>Provider Type</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Average Quality Score</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>AI Models</b>", tickangle=35, row=2, col=2)
        fig.update_yaxes(title_text="<b>Efficiency (chars/sec)</b>", row=2, col=2)
        
        # Save with enterprise-grade styling
        dashboard_file = "enterprise_model_dashboard.html"
        
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={
                'displayModeBar': True, 
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ai_model_dashboard',
                    'height': 1100,
                    'width': 1600,
                    'scale': 2
                }
            }
        )
        
        # Enterprise-grade CSS
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            margin: 0;
            padding: 24px;
            min-height: 100vh;
            color: #1e293b;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .plotly-graph-div {
            background: white;
            border-radius: 16px;
            box-shadow: 
                0 20px 25px -5px rgba(0, 0, 0, 0.1),
                0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border: 1px solid #e2e8f0;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        
        .js-plotly-plot .plotly .modebar {
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn {
            color: #64748b !important;
            transition: all 0.2s ease !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn:hover {
            background: #f1f5f9 !important;
            color: #1e293b !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn.active {
            background: #3b82f6 !important;
            color: white !important;
        }
        
        /* Custom scrollbar for better aesthetics */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        </style>
        """
        
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Enterprise dashboard created: {dashboard_file}")
        return dashboard_file
    
    def open_enterprise_dashboard(self):
        """Create and open the ultimate enterprise dashboard"""
        print("🏢 CREATING ULTIMATE ENTERPRISE DASHBOARD")
        print("=" * 55)
        
        dashboard_file = self.create_enterprise_dashboard()
        
        if dashboard_file:
            try:
                abs_path = os.path.abspath(dashboard_file)
                webbrowser.open(f'file://{abs_path}')
                print(f"🚀 Opened: {dashboard_file}")
            except Exception as e:
                print(f"❌ Failed to open dashboard: {e}")
            
            # Print enterprise summary
            gemini_count = len([r for r in self.all_results if r['provider'] == 'Gemini'])
            ollama_count = len([r for r in self.all_results if r['provider'] == 'Ollama'])
            
            print(f"\n✨ ENTERPRISE-GRADE FEATURES:")
            print("=" * 45)
            print("🎨 Clean, modern design with enterprise color system")
            print("📊 Perfect data hierarchy and visual clarity")
            print("🔤 Inter font family for professional consistency")
            print("💫 Subtle animations and micro-interactions")
            print("📱 Fully responsive with optimal spacing")
            print("🏢 Presentation-ready for executives")
            print("🎯 Interactive elements with hover states")
            print("📈 Print-friendly with high-DPI export")
            print()
            print(f"📊 COMPREHENSIVE MODEL ANALYSIS:")
            print(f"   • {gemini_count} Cloud models (Google Gemini)")
            print(f"   • {ollama_count} Local models (Ollama)")
            print(f"   • {len(self.all_results)} total models analyzed")
            print(f"   • Enterprise-grade visualizations")
            print(f"   • Professional UI/UX implementation")
        
        return dashboard_file

def main():
    """Main function"""
    dashboard = UltimateProfessionalDashboard()
    return dashboard.open_enterprise_dashboard()

if __name__ == "__main__":
    main()