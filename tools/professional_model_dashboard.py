#!/usr/bin/env python3
"""
Professional Model Comparison Dashboard
Clean, modern UI/UX with clear data presentation and professional design
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

class ProfessionalModelDashboard:
    """Create professional, clean model comparison dashboard"""
    
    def __init__(self):
        """Initialize with modern design system"""
        # Professional color palette
        self.colors = {
            'primary': '#1E3A8A',      # Deep Blue
            'secondary': '#F59E0B',    # Amber
            'success': '#10B981',      # Emerald
            'warning': '#F59E0B',      # Amber
            'danger': '#EF4444',       # Red
            'gemini': '#4285F4',       # Google Blue
            'ollama': '#FF6B35',       # Orange
            'background': '#FFFFFF',   # White
            'surface': '#F8FAFC',      # Light Gray
            'text_primary': '#1F2937', # Dark Gray
            'text_secondary': '#6B7280' # Medium Gray
        }
        
        # Typography system
        self.fonts = {
            'primary': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'mono': 'JetBrains Mono, Consolas, monospace'
        }
        
        self.load_all_data()
        
    def load_all_data(self):
        """Load and process all model data"""
        self.all_results = []
        
        # Load Gemini results
        try:
            with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
                gemini_data = json.load(f)
                gemini_results = gemini_data.get('analysis_results', [])
                
                for result in gemini_results:
                    if result['status'] == 'success':
                        # Clean model name
                        model_name = result['model'].split('/')[-1]
                        model_name = model_name.replace('gemini-', '').replace('models/', '')
                        if len(model_name) > 18:
                            model_name = model_name[:15] + '...'
                        
                        self.all_results.append({
                            'model': model_name,
                            'full_name': result['model'],
                            'provider': 'Gemini',
                            'category': 'Cloud',
                            'analysis_length': result['analysis_length'],
                            'response_time': result['response_time'],
                            'cost_estimate': (result['analysis_length'] / 1000) * 0.002,
                            'efficiency': result['analysis_length'] / result['response_time'],
                            'quality_score': min(result['analysis_length'] / 100, 100)
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
                        'quality_score': min(result['analysis_length'] / 100, 100)
                    })
            print(f"✅ Loaded {len([r for r in self.all_results if r['provider'] == 'Ollama'])} Ollama models")
        except FileNotFoundError:
            print("❌ Ollama data not found")
    
    def create_professional_dashboard(self):
        """Create clean, professional dashboard with modern UI/UX"""
        print("🎨 Creating professional dashboard with modern UI/UX...")
        
        if not self.all_results:
            print("❌ No data available")
            return None
        
        # Create clean layout with proper hierarchy
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '<b style="font-size:18px; color:#1F2937; font-family:Inter;">Model Performance Overview</b>',
                '<b style="font-size:18px; color:#1F2937; font-family:Inter;">Cost vs Quality Analysis</b>',
                '<b style="font-size:18px; color:#1F2937; font-family:Inter;">Provider Comparison</b>',
                '<b style="font-size:18px; color:#1F2937; font-family:Inter;">Efficiency Metrics</b>'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.20
        )
        
        # 1. Model Performance Overview - Clean horizontal bars
        top_models = sorted(self.all_results, key=lambda x: x['analysis_length'], reverse=True)[:12]
        
        models = [r['model'] for r in top_models]
        lengths = [r['analysis_length'] for r in top_models]
        providers = [r['provider'] for r in top_models]
        
        colors = [self.colors['gemini'] if p == 'Gemini' else self.colors['ollama'] for p in providers]
        
        fig.add_trace(go.Bar(
            y=models,
            x=lengths,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1),
                opacity=0.85
            ),
            text=[f'{l:,}' for l in lengths],
            textposition='inside',
            textfont=dict(size=11, color='white', family=self.fonts['primary'], weight='bold'),
            hovertemplate="<b style='font-family:Inter;'>%{y}</b><br>" +
                         "Analysis Length: <b>%{x:,}</b> characters<br>" +
                         "Provider: <b>%{customdata}</b><br><extra></extra>",
            customdata=providers,
            showlegend=False
        ), row=1, col=1)
        
        # 2. Cost vs Quality Analysis - Professional scatter
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
                hovertemplate="<b style='font-family:Inter;'>%{text}</b><br>" +
                             "Cost: <b>$%{x:.4f}</b><br>" +
                             "Quality Score: <b>%{y:.1f}</b><br>" +
                             "Type: <b>Cloud</b><extra></extra>"
            ), row=1, col=2)
        
        if ollama_data:
            fig.add_trace(go.Scatter(
                x=[0 for _ in ollama_data],  # Free models
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
                hovertemplate="<b style='font-family:Inter;'>%{text}</b><br>" +
                             "Cost: <b>FREE</b><br>" +
                             "Quality Score: <b>%{y:.1f}</b><br>" +
                             "Type: <b>Local</b><extra></extra>"
            ), row=1, col=2)
        
        # 3. Provider Comparison - Clean grouped bars
        provider_stats = {}
        for provider in ['Gemini', 'Ollama']:
            provider_models = [r for r in self.all_results if r['provider'] == provider]
            if provider_models:
                provider_stats[provider] = {
                    'count': len(provider_models),
                    'avg_quality': sum(r['quality_score'] for r in provider_models) / len(provider_models),
                    'avg_speed': sum(r['response_time'] for r in provider_models) / len(provider_models),
                    'total_cost': sum(r['cost_estimate'] for r in provider_models)
                }
        
        providers = list(provider_stats.keys())
        model_counts = [provider_stats[p]['count'] for p in providers]
        avg_qualities = [provider_stats[p]['avg_quality'] for p in providers]
        
        # Model count bars
        fig.add_trace(go.Bar(
            x=providers,
            y=model_counts,
            name='Model Count',
            marker=dict(
                color=[self.colors['gemini'], self.colors['ollama']],
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            text=model_counts,
            textposition='outside',
            textfont=dict(size=14, color=self.colors['text_primary'], family=self.fonts['primary'], weight='bold'),
            hovertemplate="<b style='font-family:Inter;'>%{x}</b><br>" +
                         "Models Available: <b>%{y}</b><br><extra></extra>",
            showlegend=False
        ), row=2, col=1)
        
        # 4. Efficiency Metrics - Top performers
        top_efficient = sorted(self.all_results, key=lambda x: x['efficiency'], reverse=True)[:8]
        
        eff_models = [r['model'] for r in top_efficient]
        eff_scores = [r['efficiency'] for r in top_efficient]
        eff_providers = [r['provider'] for r in top_efficient]
        eff_colors = [self.colors['gemini'] if p == 'Gemini' else self.colors['ollama'] for p in eff_providers]
        
        fig.add_trace(go.Bar(
            x=eff_models,
            y=eff_scores,
            marker=dict(
                color=eff_colors,
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            text=[f'{s:.0f}' for s in eff_scores],
            textposition='outside',
            textfont=dict(size=12, color=self.colors['text_primary'], family=self.fonts['primary']),
            hovertemplate="<b style='font-family:Inter;'>%{x}</b><br>" +
                         "Efficiency: <b>%{y:.0f}</b> chars/sec<br>" +
                         "Provider: <b>%{customdata}</b><br><extra></extra>",
            customdata=eff_providers,
            showlegend=False
        ), row=2, col=2)
        
        # Professional layout with modern design
        fig.update_layout(
            title=dict(
                text="<b style='font-size:28px; color:#1F2937; font-family:Inter;'>AI Model Performance Dashboard</b><br>" +
                     f"<span style='color:#6B7280; font-size:16px; font-family:Inter;'>" +
                     f"Comprehensive Analysis • {len(self.all_results)} Models Evaluated • Professional Grade</span>",
                x=0.5,
                y=0.95,
                xanchor='center'
            ),
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14, family=self.fonts['primary'], color=self.colors['text_primary']),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#E5E7EB',
                borderwidth=1,
                itemsizing='constant'
            ),
            template="plotly_white",
            font=dict(size=12, family=self.fonts['primary'], color=self.colors['text_primary']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            margin=dict(t=150, b=80, l=100, r=100)
        )
        
        # Clean, professional axis styling
        fig.update_xaxes(
            title_font=dict(size=14, color=self.colors['text_primary'], family=self.fonts['primary']),
            tickfont=dict(size=11, color=self.colors['text_secondary'], family=self.fonts['primary']),
            gridcolor='#F3F4F6',
            gridwidth=1,
            showline=True,
            linecolor='#D1D5DB',
            linewidth=1,
            zeroline=False,
            automargin=True
        )
        
        fig.update_yaxes(
            title_font=dict(size=14, color=self.colors['text_primary'], family=self.fonts['primary']),
            tickfont=dict(size=11, color=self.colors['text_secondary'], family=self.fonts['primary']),
            gridcolor='#F3F4F6',
            gridwidth=1,
            showline=True,
            linecolor='#D1D5DB',
            linewidth=1,
            zeroline=False,
            automargin=True
        )
        
        # Specific axis labels with clear hierarchy
        fig.update_xaxes(title_text="<b>Analysis Output (Characters)</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>AI Models</b>", row=1, col=1)
        fig.update_xaxes(title_text="<b>Cost per Analysis (USD)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Quality Score</b>", row=1, col=2)
        fig.update_xaxes(title_text="<b>Provider</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Available Models</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>AI Models</b>", tickangle=45, row=2, col=2)
        fig.update_yaxes(title_text="<b>Efficiency (chars/sec)</b>", row=2, col=2)
        
        # Save with professional naming
        dashboard_file = "professional_model_dashboard.html"
        
        # Custom HTML with additional styling
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
        )
        
        # Add custom CSS for enhanced professional appearance
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .plotly-graph-div {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #e5e7eb;
            overflow: hidden;
        }
        
        .js-plotly-plot .plotly .modebar {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 8px !important;
            border: 1px solid #e5e7eb !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn {
            color: #6b7280 !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn:hover {
            background: #f3f4f6 !important;
            color: #1f2937 !important;
        }
        </style>
        """
        
        # Insert custom CSS into HTML
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Professional dashboard created: {dashboard_file}")
        return dashboard_file
    
    def create_summary_cards(self):
        """Create additional summary dashboard with cards layout"""
        print("📊 Creating summary cards dashboard...")
        
        # Calculate key metrics
        best_overall = max(self.all_results, key=lambda x: x['analysis_length'])
        best_local = max([r for r in self.all_results if r['provider'] == 'Ollama'], 
                        key=lambda x: x['analysis_length']) if any(r['provider'] == 'Ollama' for r in self.all_results) else None
        fastest = min(self.all_results, key=lambda x: x['response_time'])
        most_efficient = max(self.all_results, key=lambda x: x['efficiency'])
        
        gemini_count = len([r for r in self.all_results if r['provider'] == 'Gemini'])
        ollama_count = len([r for r in self.all_results if r['provider'] == 'Ollama'])
        
        # Create cards-style layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                '<b style="color:#10B981;">🏆 Best Overall Performance</b>',
                '<b style="color:#F59E0B;">💰 Best Value (Free)</b>',
                '<b style="color:#EF4444;">⚡ Fastest Response</b>',
                '<b style="color:#8B5CF6;">📊 Model Coverage</b>',
                '<b style="color:#06B6D4;">🚀 Efficiency Leader</b>',
                '<b style="color:#84CC16;">💡 Key Insights</b>'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "indicator"}, {"type": "table"}]
            ],
            vertical_spacing=0.25,
            horizontal_spacing=0.15
        )
        
        # 1. Best Overall Performance Card
        fig.add_trace(go.Indicator(
            mode="number+gauge+delta",
            value=best_overall['analysis_length'],
            title={"text": f"<b>{best_overall['model']}</b><br><span style='font-size:14px;'>{best_overall['provider']} Model</span>"},
            number={'suffix': " chars", 'font': {'size': 24, 'color': '#10B981'}},
            gauge={'axis': {'range': [None, 10000]}, 'bar': {'color': '#10B981'}, 'bgcolor': '#F0FDF4'},
            domain={'row': 0, 'column': 0}
        ), row=1, col=1)
        
        # 2. Best Value Card
        if best_local:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=best_local['analysis_length'],
                title={"text": f"<b>{best_local['model']}</b><br><span style='font-size:14px;'>FREE Local Model</span>"},
                number={'suffix': " chars", 'font': {'size': 24, 'color': '#F59E0B'}},
                gauge={'axis': {'range': [None, 5000]}, 'bar': {'color': '#F59E0B'}, 'bgcolor': '#FFFBEB'},
                domain={'row': 0, 'column': 1}
            ), row=1, col=2)
        
        # 3. Fastest Response Card
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=fastest['response_time'],
            title={"text": f"<b>{fastest['model']}</b><br><span style='font-size:14px;'>Speed Champion</span>"},
            number={'suffix': "s", 'font': {'size': 24, 'color': '#EF4444'}},
            gauge={'axis': {'range': [0, 20]}, 'bar': {'color': '#EF4444'}, 'bgcolor': '#FEF2F2'},
            domain={'row': 0, 'column': 2}
        ), row=1, col=3)
        
        # 4. Model Coverage Bar
        fig.add_trace(go.Bar(
            x=['Cloud Models', 'Local Models'],
            y=[gemini_count, ollama_count],
            marker=dict(color=[self.colors['gemini'], self.colors['ollama']], opacity=0.8),
            text=[gemini_count, ollama_count],
            textposition='outside',
            textfont=dict(size=16, color=self.colors['text_primary'], family=self.fonts['primary']),
            showlegend=False
        ), row=2, col=1)
        
        # 5. Efficiency Leader Card
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=most_efficient['efficiency'],
            title={"text": f"<b>{most_efficient['model']}</b><br><span style='font-size:14px;'>Most Efficient</span>"},
            number={'suffix': " c/s", 'font': {'size': 24, 'color': '#06B6D4'}},
            gauge={'axis': {'range': [0, 2000]}, 'bar': {'color': '#06B6D4'}, 'bgcolor': '#F0F9FF'},
            domain={'row': 1, 'column': 4}
        ), row=2, col=2)
        
        # 6. Key Insights Table
        insights_data = [
            ['Total Models Tested', f'{len(self.all_results)}'],
            ['Cloud vs Local Split', f'{gemini_count}:{ollama_count}'],
            ['Best Quality Score', f'{best_overall["quality_score"]:.1f}/100'],
            ['Average Response Time', f'{sum(r["response_time"] for r in self.all_results)/len(self.all_results):.1f}s'],
            ['Total Analysis Capacity', f'{sum(r["analysis_length"] for r in self.all_results):,} chars']
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='#1F2937',
                align='center',
                font=dict(size=14, color='white', family=self.fonts['primary']),
                height=40
            ),
            cells=dict(
                values=list(zip(*insights_data)),
                fill_color=['#F8FAFC', '#FFFFFF'],
                align=['left', 'center'],
                font=dict(size=13, color=self.colors['text_primary'], family=self.fonts['primary']),
                height=35
            )
        ), row=2, col=3)
        
        # Professional layout for cards
        fig.update_layout(
            title=dict(
                text="<b style='font-size:26px; color:#1F2937; font-family:Inter;'>Executive Summary Dashboard</b><br>" +
                     "<span style='color:#6B7280; font-size:15px; font-family:Inter;'>Key Performance Indicators & Insights</span>",
                x=0.5,
                y=0.95
            ),
            height=800,
            showlegend=False,
            template="plotly_white",
            font=dict(size=12, family=self.fonts['primary'], color=self.colors['text_primary']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            margin=dict(t=120, b=60, l=80, r=80)
        )
        
        # Save summary dashboard
        summary_file = "executive_summary_dashboard.html"
        
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={'displayModeBar': False}
        )
        
        # Add the same professional CSS
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .plotly-graph-div {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e5e7eb;
            overflow: hidden;
        }
        </style>
        """
        
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Executive summary created: {summary_file}")
        return summary_file
    
    def open_professional_dashboards(self):
        """Create and open professional dashboards"""
        print("🎨 CREATING PROFESSIONAL UI/UX DASHBOARDS")
        print("=" * 50)
        
        # Create main dashboard
        main_dashboard = self.create_professional_dashboard()
        
        # Create summary dashboard
        summary_dashboard = self.create_summary_cards()
        
        if main_dashboard and summary_dashboard:
            # Open both dashboards
            try:
                abs_path1 = os.path.abspath(main_dashboard)
                abs_path2 = os.path.abspath(summary_dashboard)
                webbrowser.open(f'file://{abs_path1}')
                webbrowser.open(f'file://{abs_path2}')
                print(f"🚀 Opened: {main_dashboard}")
                print(f"🚀 Opened: {summary_dashboard}")
            except Exception as e:
                print(f"❌ Failed to open dashboards: {e}")
            
            # Print professional summary
            gemini_count = len([r for r in self.all_results if r['provider'] == 'Gemini'])
            ollama_count = len([r for r in self.all_results if r['provider'] == 'Ollama'])
            
            print(f"\n✨ PROFESSIONAL DESIGN FEATURES:")
            print("=" * 45)
            print("🎨 Modern, clean design with professional color palette")
            print("📊 Clear data hierarchy and improved readability")
            print("🔤 Professional typography with Inter font family")
            print("💫 Subtle gradients and shadows for depth")
            print("📱 Responsive layout with optimal spacing")
            print("🎯 Executive summary cards for key insights")
            print("🏢 Enterprise-ready presentation quality")
            print()
            print(f"📈 COMPREHENSIVE ANALYSIS:")
            print(f"   • {gemini_count} Cloud models (Gemini)")
            print(f"   • {ollama_count} Local models (Ollama)")
            print(f"   • {len(self.all_results)} total models analyzed")
            print(f"   • Professional-grade visualizations")
        
        return [main_dashboard, summary_dashboard]

def main():
    """Main function"""
    dashboard = ProfessionalModelDashboard()
    return dashboard.open_professional_dashboards()

if __name__ == "__main__":
    main()