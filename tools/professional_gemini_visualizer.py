#!/usr/bin/env python3
"""
Professional Gemini Model Visualizer
Clean, modern design focused on data clarity and professional presentation
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

class ProfessionalGeminiVisualizer:
    """Create professional Gemini model analysis dashboards"""
    
    def __init__(self):
        """Initialize with professional design system"""
        # Professional color palette
        self.colors = {
            'primary': '#1565C0',      # Deep Blue
            'secondary': '#43A047',    # Green
            'accent': '#FB8C00',       # Orange
            'success': '#00ACC1',      # Cyan
            'warning': '#FFB300',      # Amber
            'error': '#E53935',        # Red
            'gemini_1': '#4285F4',     # Google Blue
            'gemini_2': '#34A853',     # Google Green
            'gemini_3': '#FBBC04',     # Google Yellow
            'gemini_4': '#EA4335',     # Google Red
            'background': '#FAFAFA',   # Light Gray
            'surface': '#FFFFFF',      # White
            'text_primary': '#212121', # Dark Gray
            'text_secondary': '#757575', # Medium Gray
            'border': '#E0E0E0'        # Light Border
        }
        
        # Typography system
        self.fonts = {
            'primary': 'Roboto, -apple-system, BlinkMacSystemFont, sans-serif',
            'secondary': 'Roboto Mono, Monaco, monospace'
        }
        
        self.load_gemini_data()
        
    def load_gemini_data(self):
        """Load and process Gemini analysis data"""
        try:
            with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
                self.data = json.load(f)
                
            self.analysis_results = []
            for result in self.data.get('analysis_results', []):
                if result['status'] == 'success':
                    # Clean model name for display
                    model_name = result['model'].split('/')[-1]
                    model_name = model_name.replace('gemini-', '').replace('models/', '')
                    
                    # Categorize by generation
                    if '1.0' in model_name or '1.5' in model_name:
                        generation = 'Gen 1.x'
                        color = self.colors['gemini_1']
                    elif '2.0' in model_name:
                        generation = 'Gen 2.0'
                        color = self.colors['gemini_2']
                    elif '2.5' in model_name:
                        generation = 'Gen 2.5'
                        color = self.colors['gemini_3']
                    elif 'gemma' in model_name.lower():
                        generation = 'Gemma'
                        color = self.colors['gemini_4']
                    else:
                        generation = 'Other'
                        color = self.colors['accent']
                    
                    self.analysis_results.append({
                        'model': model_name,
                        'full_name': result['model'],
                        'generation': generation,
                        'analysis_length': result['analysis_length'],
                        'response_time': result['response_time'],
                        'efficiency': result['analysis_length'] / result['response_time'],
                        'cost_estimate': (result['analysis_length'] / 1000) * 0.002,
                        'quality_score': min(result['analysis_length'] / 100, 100),
                        'color': color
                    })
            
            print(f"✅ Loaded {len(self.analysis_results)} successful Gemini model analyses")
            
        except FileNotFoundError:
            print("❌ Gemini analysis data not found")
            self.analysis_results = []
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.analysis_results = []
    
    def create_main_dashboard(self):
        """Create main professional Gemini dashboard"""
        print("🎨 Creating professional main Gemini dashboard...")
        
        if not self.analysis_results:
            return None
        
        # Create clean, focused layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '<b style="font-size:16px; color:#212121;">Model Performance by Generation</b>',
                '<b style="font-size:16px; color:#212121;">Efficiency vs Quality Matrix</b>',
                '<b style="font-size:16px; color:#212121;">Response Time Distribution</b>',
                '<b style="font-size:16px; color:#212121;">Cost-Benefit Analysis</b>'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "bar"}]],
            vertical_spacing=0.20,
            horizontal_spacing=0.15
        )
        
        # 1. Performance by Generation - Clean grouped bars
        generations = {}
        for result in self.analysis_results:
            gen = result['generation']
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(result)
        
        gen_names = list(generations.keys())
        avg_lengths = [sum(r['analysis_length'] for r in generations[gen]) / len(generations[gen]) for gen in gen_names]
        model_counts = [len(generations[gen]) for gen in gen_names]
        gen_colors = [self.colors['gemini_1'], self.colors['gemini_2'], self.colors['gemini_3'], self.colors['gemini_4']][:len(gen_names)]
        
        fig.add_trace(go.Bar(
            x=gen_names,
            y=avg_lengths,
            marker=dict(
                color=gen_colors,
                opacity=0.85,
                line=dict(color='white', width=2)
            ),
            text=[f'{int(length):,}' for length in avg_lengths],
            textposition='outside',
            textfont=dict(size=12, color=self.colors['text_primary'], family=self.fonts['primary']),
            hovertemplate="<b style='font-family:Roboto;'>%{x}</b><br>" +
                         "Avg Length: <b>%{y:,.0f}</b> chars<br>" +
                         "Models: <b>%{customdata}</b><br><extra></extra>",
            customdata=model_counts,
            showlegend=False
        ), row=1, col=1)
        
        # 2. Efficiency vs Quality Matrix - Professional scatter
        for gen in gen_names:
            gen_data = generations[gen]
            fig.add_trace(go.Scatter(
                x=[r['efficiency'] for r in gen_data],
                y=[r['quality_score'] for r in gen_data],
                mode='markers',
                name=gen,
                marker=dict(
                    size=12,
                    color=gen_data[0]['color'],
                    opacity=0.8,
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                text=[r['model'] for r in gen_data],
                hovertemplate="<b style='font-family:Roboto;'>%{text}</b><br>" +
                             "Efficiency: <b>%{x:.0f}</b> chars/sec<br>" +
                             "Quality: <b>%{y:.1f}</b>/100<br>" +
                             "Generation: <b>" + gen + "</b><extra></extra>"
            ), row=1, col=2)
        
        # 3. Response Time Distribution - Clean box plots
        for gen in gen_names:
            gen_data = generations[gen]
            fig.add_trace(go.Box(
                y=[r['response_time'] for r in gen_data],
                name=gen,
                marker=dict(color=gen_data[0]['color'], opacity=0.7),
                boxpoints='all',
                jitter=0.3,
                pointpos=0,
                line=dict(width=2),
                showlegend=False
            ), row=2, col=1)
        
        # 4. Cost-Benefit Analysis - Top performers
        top_value = sorted(self.analysis_results, key=lambda x: x['quality_score'] / max(x['cost_estimate'], 0.001), reverse=True)[:10]
        
        fig.add_trace(go.Bar(
            x=[r['model'][:15] + '...' if len(r['model']) > 15 else r['model'] for r in top_value],
            y=[r['quality_score'] / max(r['cost_estimate'], 0.001) for r in top_value],
            marker=dict(
                color=[r['color'] for r in top_value],
                opacity=0.85,
                line=dict(color='white', width=2)
            ),
            text=[f'{r["quality_score"] / max(r["cost_estimate"], 0.001):.0f}' for r in top_value],
            textposition='outside',
            textfont=dict(size=11, color=self.colors['text_primary'], family=self.fonts['primary']),
            hovertemplate="<b style='font-family:Roboto;'>%{customdata}</b><br>" +
                         "Value Score: <b>%{y:.0f}</b><br>" +
                         "Quality: <b>%{meta[0]:.1f}</b>/100<br>" +
                         "Cost: <b>$%{meta[1]:.4f}</b><br><extra></extra>",
            customdata=[r['model'] for r in top_value],
            meta=[[r['quality_score'], r['cost_estimate']] for r in top_value],
            showlegend=False
        ), row=2, col=2)
        
        # Professional layout
        fig.update_layout(
            title=dict(
                text="<b style='font-size:24px; color:#212121; font-family:Roboto;'>Gemini Model Analysis Dashboard</b><br>" +
                     f"<span style='color:#757575; font-size:14px; font-family:Roboto;'>" +
                     f"{len(self.analysis_results)} Models Analyzed • Professional Grade Assessment</span>",
                x=0.5,
                y=0.95
            ),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12, family=self.fonts['primary'], color=self.colors['text_primary']),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=self.colors['border'],
                borderwidth=1
            ),
            template="plotly_white",
            font=dict(size=11, family=self.fonts['primary'], color=self.colors['text_primary']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            margin=dict(t=130, b=70, l=80, r=80)
        )
        
        # Clean axis styling
        fig.update_xaxes(
            title_font=dict(size=12, color=self.colors['text_primary'], family=self.fonts['primary']),
            tickfont=dict(size=10, color=self.colors['text_secondary'], family=self.fonts['primary']),
            gridcolor='#F5F5F5',
            gridwidth=1,
            showline=True,
            linecolor=self.colors['border'],
            linewidth=1,
            automargin=True
        )
        
        fig.update_yaxes(
            title_font=dict(size=12, color=self.colors['text_primary'], family=self.fonts['primary']),
            tickfont=dict(size=10, color=self.colors['text_secondary'], family=self.fonts['primary']),
            gridcolor='#F5F5F5',
            gridwidth=1,
            showline=True,
            linecolor=self.colors['border'],
            linewidth=1,
            automargin=True
        )
        
        # Axis labels
        fig.update_xaxes(title_text="<b>Model Generation</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>Avg Analysis Length</b>", row=1, col=1)
        fig.update_xaxes(title_text="<b>Efficiency (chars/sec)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Quality Score</b>", row=1, col=2)
        fig.update_xaxes(title_text="<b>Model Generation</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Response Time (seconds)</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>Models</b>", tickangle=45, row=2, col=2)
        fig.update_yaxes(title_text="<b>Value Score</b>", row=2, col=2)
        
        # Save with professional styling
        dashboard_file = "professional_gemini_dashboard.html"
        
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        # Professional CSS
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
        
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
            margin: 0;
            padding: 24px;
            min-height: 100vh;
            color: #212121;
        }
        
        .plotly-graph-div {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1), 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
            overflow: hidden;
        }
        
        .js-plotly-plot .plotly .modebar {
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 4px !important;
            border: 1px solid #e0e0e0 !important;
        }
        </style>
        """
        
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Professional Gemini dashboard created: {dashboard_file}")
        return dashboard_file
    
    def create_detailed_comparison(self):
        """Create detailed model comparison table"""
        print("📊 Creating detailed comparison dashboard...")
        
        if not self.analysis_results:
            return None
        
        # Sort by quality score
        sorted_models = sorted(self.analysis_results, key=lambda x: x['quality_score'], reverse=True)
        
        # Create detailed table
        fig = go.Figure()
        
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    '<b>Rank</b>',
                    '<b>Model Name</b>',
                    '<b>Generation</b>',
                    '<b>Quality Score</b>',
                    '<b>Analysis Length</b>',
                    '<b>Response Time</b>',
                    '<b>Efficiency</b>',
                    '<b>Cost Estimate</b>'
                ],
                fill_color=self.colors['primary'],
                align='center',
                font=dict(size=13, color='white', family=self.fonts['primary']),
                height=50,
                line=dict(color='white', width=2)
            ),
            cells=dict(
                values=[
                    list(range(1, len(sorted_models) + 1)),
                    [model['model'] for model in sorted_models],
                    [model['generation'] for model in sorted_models],
                    [f"{model['quality_score']:.1f}/100" for model in sorted_models],
                    [f"{model['analysis_length']:,}" for model in sorted_models],
                    [f"{model['response_time']:.2f}s" for model in sorted_models],
                    [f"{model['efficiency']:.0f} c/s" for model in sorted_models],
                    [f"${model['cost_estimate']:.4f}" for model in sorted_models]
                ],
                fill_color=[
                    ['#FAFAFA' if i % 2 == 0 else '#FFFFFF' for i in range(len(sorted_models))]
                ] * 8,
                align=['center', 'left', 'center', 'center', 'right', 'center', 'center', 'right'],
                font=dict(size=11, color=self.colors['text_primary'], family=self.fonts['primary']),
                height=40,
                line=dict(color=self.colors['border'], width=1)
            )
        ))
        
        fig.update_layout(
            title=dict(
                text="<b style='font-size:22px; color:#212121; font-family:Roboto;'>Detailed Model Comparison</b><br>" +
                     "<span style='color:#757575; font-size:13px; font-family:Roboto;'>Complete Performance Rankings</span>",
                x=0.5,
                y=0.95
            ),
            height=600,
            font=dict(family=self.fonts['primary']),
            paper_bgcolor=self.colors['background'],
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Save detailed table
        table_file = "professional_model_comparison_table.html"
        
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={'displayModeBar': False}
        )
        
        # Add professional CSS
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            background: #fafafa;
            margin: 0;
            padding: 24px;
            color: #212121;
        }
        
        .plotly-graph-div {
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
        }
        </style>
        """
        
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Professional comparison table created: {table_file}")
        return table_file
    
    def create_executive_summary(self):
        """Create executive summary with key insights"""
        print("📈 Creating executive summary dashboard...")
        
        if not self.analysis_results:
            return None
        
        # Calculate key metrics
        best_overall = max(self.analysis_results, key=lambda x: x['quality_score'])
        fastest = min(self.analysis_results, key=lambda x: x['response_time'])
        most_efficient = max(self.analysis_results, key=lambda x: x['efficiency'])
        best_value = min(self.analysis_results, key=lambda x: x['cost_estimate'])
        
        # Generation statistics
        gen_stats = {}
        for result in self.analysis_results:
            gen = result['generation']
            if gen not in gen_stats:
                gen_stats[gen] = {'count': 0, 'avg_quality': 0, 'models': []}
            gen_stats[gen]['count'] += 1
            gen_stats[gen]['models'].append(result)
        
        for gen in gen_stats:
            gen_stats[gen]['avg_quality'] = sum(m['quality_score'] for m in gen_stats[gen]['models']) / len(gen_stats[gen]['models'])
        
        # Create summary layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '<b style="color:#1565C0;">🏆 Performance Champions</b>',
                '<b style="color:#43A047;">📊 Generation Overview</b>',
                '<b style="color:#FB8C00;">⚡ Key Metrics</b>',
                '<b style="color:#00ACC1;">💡 Executive Insights</b>'
            ],
            specs=[
                [{"type": "table"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "table"}]
            ],
            vertical_spacing=0.25,
            horizontal_spacing=0.15
        )
        
        # 1. Performance Champions Table
        champions_data = [
            ['🏆 Best Overall', best_overall['model'], f"{best_overall['quality_score']:.1f}/100"],
            ['⚡ Fastest Response', fastest['model'], f"{fastest['response_time']:.2f}s"],
            ['🚀 Most Efficient', most_efficient['model'], f"{most_efficient['efficiency']:.0f} c/s"],
            ['💰 Best Value', best_value['model'], f"${best_value['cost_estimate']:.4f}"]
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Category</b>', '<b>Model</b>', '<b>Score</b>'],
                fill_color=self.colors['primary'],
                align='center',
                font=dict(size=13, color='white', family=self.fonts['primary']),
                height=45
            ),
            cells=dict(
                values=list(zip(*champions_data)),
                fill_color=[['#E3F2FD', '#FFFFFF', '#E3F2FD', '#FFFFFF']],
                align=['center', 'left', 'center'],
                font=dict(size=12, color=self.colors['text_primary'], family=self.fonts['primary']),
                height=45
            )
        ), row=1, col=1)
        
        # 2. Generation Overview
        gen_names = list(gen_stats.keys())
        gen_counts = [gen_stats[gen]['count'] for gen in gen_names]
        gen_colors = [self.colors['gemini_1'], self.colors['gemini_2'], self.colors['gemini_3'], self.colors['gemini_4']][:len(gen_names)]
        
        fig.add_trace(go.Bar(
            x=gen_names,
            y=gen_counts,
            marker=dict(color=gen_colors, opacity=0.8),
            text=gen_counts,
            textposition='outside',
            textfont=dict(size=14, color=self.colors['text_primary'], family=self.fonts['primary']),
            showlegend=False
        ), row=1, col=2)
        
        # 3. Key Metrics Indicator
        avg_quality = sum(r['quality_score'] for r in self.analysis_results) / len(self.analysis_results)
        
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=avg_quality,
            title={"text": "<b>Average Quality Score</b><br><span style='font-size:12px;'>Across All Models</span>"},
            number={'suffix': "/100", 'font': {'size': 28, 'color': self.colors['success']}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': self.colors['success']}, 'bgcolor': '#E8F5E8'},
        ), row=2, col=1)
        
        # 4. Executive Insights
        insights_data = [
            ['Total Models Analyzed', f'{len(self.analysis_results)}'],
            ['Best Generation', f'{max(gen_stats.keys(), key=lambda x: gen_stats[x]["avg_quality"])}'],
            ['Quality Range', f'{min(r["quality_score"] for r in self.analysis_results):.1f} - {max(r["quality_score"] for r in self.analysis_results):.1f}'],
            ['Speed Range', f'{min(r["response_time"] for r in self.analysis_results):.1f}s - {max(r["response_time"] for r in self.analysis_results):.1f}s'],
            ['Total Cost for All', f'${sum(r["cost_estimate"] for r in self.analysis_results):.3f}']
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color=self.colors['success'],
                align='center',
                font=dict(size=13, color='white', family=self.fonts['primary']),
                height=40
            ),
            cells=dict(
                values=list(zip(*insights_data)),
                fill_color=[['#F1F8E9', '#FFFFFF'] * 3],
                align=['left', 'center'],
                font=dict(size=12, color=self.colors['text_primary'], family=self.fonts['primary']),
                height=35
            )
        ), row=2, col=2)
        
        # Professional layout
        fig.update_layout(
            title=dict(
                text="<b style='font-size:24px; color:#212121; font-family:Roboto;'>Executive Summary</b><br>" +
                     "<span style='color:#757575; font-size:14px; font-family:Roboto;'>Gemini Model Performance Overview</span>",
                x=0.5,
                y=0.95
            ),
            height=700,
            showlegend=False,
            font=dict(family=self.fonts['primary'], color=self.colors['text_primary']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            margin=dict(t=120, b=60, l=80, r=80)
        )
        
        # Save executive summary
        summary_file = "professional_executive_summary.html"
        
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={'displayModeBar': False}
        )
        
        # Professional CSS
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
            margin: 0;
            padding: 24px;
            color: #212121;
        }
        
        .plotly-graph-div {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        </style>
        """
        
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Professional executive summary created: {summary_file}")
        return summary_file
    
    def open_professional_suite(self):
        """Create and open complete professional dashboard suite"""
        print("🎨 CREATING PROFESSIONAL GEMINI DASHBOARD SUITE")
        print("=" * 55)
        
        dashboards = []
        
        # Create all dashboards
        main_dashboard = self.create_main_dashboard()
        comparison_table = self.create_detailed_comparison()
        executive_summary = self.create_executive_summary()
        
        if main_dashboard:
            dashboards.append(main_dashboard)
        if comparison_table:
            dashboards.append(comparison_table)
        if executive_summary:
            dashboards.append(executive_summary)
        
        # Open all dashboards
        for dashboard in dashboards:
            try:
                abs_path = os.path.abspath(dashboard)
                webbrowser.open(f'file://{abs_path}')
                print(f"🚀 Opened: {dashboard}")
            except Exception as e:
                print(f"❌ Failed to open {dashboard}: {e}")
        
        # Print summary
        print(f"\n✨ PROFESSIONAL DESIGN FEATURES:")
        print("=" * 45)
        print("🎨 Clean, modern design with Google Material colors")
        print("📊 Clear data hierarchy and professional typography")
        print("🔤 Roboto font family for enterprise consistency")
        print("💫 Subtle shadows and gradients for depth")
        print("📱 Responsive layout with optimal white space")
        print("🏢 Executive-ready presentation quality")
        print()
        print(f"📈 GEMINI ANALYSIS SUITE:")
        print(f"   • {len(self.analysis_results)} models analyzed")
        print(f"   • Multiple generations covered")
        print(f"   • Professional visualizations created")
        print(f"   • {len(dashboards)} dashboards generated")
        
        return dashboards

def main():
    """Main function"""
    visualizer = ProfessionalGeminiVisualizer()
    return visualizer.open_professional_suite()

if __name__ == "__main__":
    main()