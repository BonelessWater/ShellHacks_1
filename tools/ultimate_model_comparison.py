#!/usr/bin/env python3
"""
Ultimate Model Comparison Dashboard
Shows comprehensive comparison between all Gemini and Ollama models
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

class UltimateModelComparison:
    """Create ultimate comparison dashboard"""
    
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
                        self.all_results.append({
                            'model': result['model'].split('/')[-1].replace('gemini-', ''),
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
        
        # Load latest multi-Ollama results
        try:
            with open('multi_ollama_results.json', 'r') as f:
                ollama_data = json.load(f)
                ollama_results = ollama_data.get('results', [])
                
                for result in ollama_results:
                    self.all_results.append({
                        'model': result['model'],
                        'provider': 'Ollama',
                        'category': 'Local',
                        'analysis_length': result['analysis_length'],
                        'response_time': result['response_time'],
                        'cost_estimate': 0.0,  # Local is free
                        'efficiency': result.get('efficiency', result['analysis_length'] / result['response_time'])
                    })
            print(f"✅ Loaded {len([r for r in self.all_results if r['provider'] == 'Ollama'])} Ollama models")
        except FileNotFoundError:
            print("❌ Multi-Ollama data not found")
    
    def create_ultimate_dashboard(self):
        """Create the ultimate comparison dashboard"""
        print("🚀 Creating ultimate model comparison dashboard...")
        
        if not self.all_results:
            print("❌ No data available")
            return None
        
        # Create comprehensive comparison figure with improved spacing
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '<b style="font-size:16px;">🏆 All Models: Analysis Length Ranking</b>',
                '<b style="font-size:16px;">⚡ Speed vs Quality Analysis</b>',
                '<b style="font-size:16px;">💰 Cost Comparison: Cloud vs Local</b>',
                '<b style="font-size:16px;">🎯 Provider Performance Distribution</b>',
                '<b style="font-size:16px;">🚀 Efficiency Champions</b>',
                '<b style="font-size:16px;">📊 Model Categories Overview</b>',
                '<b style="font-size:16px;">🥇 Top Performers by Category</b>',
                '<b style="font-size:16px;">📈 Performance Metrics Summary</b>'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "table"}]],
            vertical_spacing=0.15,  # Increased spacing between rows
            horizontal_spacing=0.15  # Increased spacing between columns
        )
        
        # Sort all results by analysis length
        sorted_results = sorted(self.all_results, key=lambda x: x['analysis_length'], reverse=True)
        
        # Take top 15 for visibility
        top_results = sorted_results[:15]
        
        # 1. All Models Ranking - Enhanced horizontal bar chart
        models = [r['model'][:25] + '...' if len(r['model']) > 25 else r['model'] for r in top_results]  # Truncate long names
        lengths = [r['analysis_length'] for r in top_results]
        providers = [r['provider'] for r in top_results]
        
        colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in providers]
        
        fig.add_trace(go.Bar(
            y=models,  # Horizontal bar for better model name readability
            x=lengths,
            orientation='h',
            name="Analysis Length",
            marker=dict(
                color=colors, 
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'{l:,}' for l in lengths],
            textposition='inside',
            textfont=dict(size=11, color='white', family='Arial Black'),
            hovertemplate="<b>%{y}</b><br>Analysis: %{x:,} characters<br>Provider: %{customdata}<br><extra></extra>",
            customdata=providers,
            showlegend=False
        ), row=1, col=1)
        
        # 2. Speed vs Quality Scatter
        times = [r['response_time'] for r in self.all_results]
        all_lengths = [r['analysis_length'] for r in self.all_results]
        all_providers = [r['provider'] for r in self.all_results]
        all_models = [r['model'] for r in self.all_results]
        
        # Create separate traces for each provider
        for provider in ['Gemini', 'Ollama']:
            provider_data = [r for r in self.all_results if r['provider'] == provider]
            if provider_data:
                p_times = [r['response_time'] for r in provider_data]
                p_lengths = [r['analysis_length'] for r in provider_data]
                p_models = [r['model'] for r in provider_data]
                p_efficiencies = [r['efficiency'] for r in provider_data]
                
                color = '#4285F4' if provider == 'Gemini' else '#FF6B35'
                
                fig.add_trace(go.Scatter(
                    x=p_times,
                    y=p_lengths,
                    mode='markers',
                    name=f'{provider} Models',
                    marker=dict(
                        size=[max(8, min(25, eff/50)) for eff in p_efficiencies],
                        color=color,
                        opacity=0.8,
                        line=dict(color='white', width=2)
                    ),
                    text=p_models,
                    hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<br>Provider: " + provider + "<extra></extra>"
                ), row=1, col=2)
        
        # 3. Cost Comparison
        cost_comparison = {}
        for result in self.all_results:
            provider = result['provider']
            if provider not in cost_comparison:
                cost_comparison[provider] = []
            cost_comparison[provider].append(result['cost_estimate'])
        
        providers_list = list(cost_comparison.keys())
        avg_costs = [sum(costs)/len(costs) for costs in cost_comparison.values()]
        
        fig.add_trace(go.Bar(
            x=providers_list,
            y=avg_costs,
            name="Average Cost",
            marker=dict(color=['#4285F4', '#FF6B35'], line=dict(color='white', width=2)),
            text=[f'${cost:.4f}' if cost > 0 else 'FREE' for cost in avg_costs],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Avg Cost: %{text}<extra></extra>"
        ), row=2, col=1)
        
        # 4. Provider Performance Distribution
        for provider in ['Gemini', 'Ollama']:
            provider_lengths = [r['analysis_length'] for r in self.all_results if r['provider'] == provider]
            if provider_lengths:
                color = '#4285F4' if provider == 'Gemini' else '#FF6B35'
                fig.add_trace(go.Box(
                    y=provider_lengths,
                    name=provider,
                    marker_color=color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ), row=2, col=2)
        
        # 5. Efficiency Champions - Enhanced bar chart with better spacing
        top_efficient = sorted(self.all_results, key=lambda x: x['efficiency'], reverse=True)[:8]  # Reduced to 8 for better spacing
        eff_models = [r['model'][:15] + '...' if len(r['model']) > 15 else r['model'] for r in top_efficient]  # Truncate names
        eff_scores = [r['efficiency'] for r in top_efficient]
        eff_providers = [r['provider'] for r in top_efficient]
        eff_colors = ['#4285F4' if p == 'Gemini' else '#FF6B35' for p in eff_providers]
        
        fig.add_trace(go.Bar(
            x=eff_models,
            y=eff_scores,
            name="Efficiency",
            marker=dict(
                color=eff_colors, 
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'{s:.0f}' for s in eff_scores],
            textposition='outside',
            textfont=dict(size=11, color='#2C3E50'),
            hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.0f} chars/sec<br>Provider: %{customdata}<br><extra></extra>",
            customdata=eff_providers,
            showlegend=False
        ), row=3, col=1)
        
        # 6. Model Categories Pie
        category_counts = {}
        for result in self.all_results:
            category = result['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        fig.add_trace(go.Pie(
            labels=list(category_counts.keys()),
            values=list(category_counts.values()),
            name="Categories",
            marker=dict(
                colors=['#4285F4', '#FF6B35'], 
                line=dict(color='white', width=3)
            ),
            textinfo='label+percent+value',
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<br><extra></extra>",
            pull=[0.05, 0.05],  # Slight separation for visual appeal
            showlegend=False
        ), row=3, col=2)
        
        # 7. Top Performers by Category
        best_cloud = max([r for r in self.all_results if r['category'] == 'Cloud'], 
                        key=lambda x: x['analysis_length'])
        best_local = max([r for r in self.all_results if r['category'] == 'Local'], 
                        key=lambda x: x['analysis_length'])
        
        category_winners = [best_cloud, best_local]
        winner_names = [r['model'] for r in category_winners]
        winner_lengths = [r['analysis_length'] for r in category_winners]
        winner_colors = ['#4285F4', '#FF6B35']
        
        fig.add_trace(go.Bar(
            x=winner_names,
            y=winner_lengths,
            name="Category Winners",
            marker=dict(color=winner_colors, line=dict(color='white', width=2)),
            text=[f'{l:,}' for l in winner_lengths],
            textposition='outside'
        ), row=4, col=1)
        
        # 8. Summary Table
        summary_data = [
            ['🏆 Best Overall', best_cloud['model'], f"{best_cloud['analysis_length']:,} chars", f"${best_cloud['cost_estimate']:.4f}"],
            ['💰 Best Value', best_local['model'], f"{best_local['analysis_length']:,} chars", 'FREE'],
            ['⚡ Fastest', sorted(self.all_results, key=lambda x: x['response_time'])[0]['model'], 
             f"{sorted(self.all_results, key=lambda x: x['response_time'])[0]['response_time']:.2f}s", 'Variable'],
            ['🚀 Most Efficient', sorted(self.all_results, key=lambda x: x['efficiency'], reverse=True)[0]['model'],
             f"{sorted(self.all_results, key=lambda x: x['efficiency'], reverse=True)[0]['efficiency']:.0f} c/s", 'Variable']
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['🎯 Category', '🤖 Model', '📊 Performance', '💰 Cost'],
                fill_color='#2C3E50',
                align='center',
                font=dict(size=14, color='white', family='Arial Black'),
                height=40
            ),
            cells=dict(
                values=list(zip(*summary_data)),
                fill_color=[['#ECF0F1', '#FFFFFF'] * 2],
                align=['center', 'left', 'center', 'center'],
                font=dict(size=12, color='#2C3E50'),
                height=50
            )
        ), row=4, col=2)
        
        # Update layout with enhanced UI/UX
        fig.update_layout(
            title=dict(
                text="<b style='font-size:28px;'>🚀 Ultimate AI Model Comparison Dashboard</b><br>" +
                     f"<span style='color: #7F8C8D; font-size:16px;'>{len([r for r in self.all_results if r['provider'] == 'Gemini'])} Gemini Models vs " +
                     f"{len([r for r in self.all_results if r['provider'] == 'Ollama'])} Local Ollama Models • ShellHacks 2025</span>",
                x=0.5,
                y=0.98,
                font=dict(color='#2C3E50', family='Arial Black')
            ),
            height=1800,  # Increased height for better spacing
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#BDC3C7',
                borderwidth=1
            ),
            template="plotly_white",
            font=dict(size=12, family="Arial, sans-serif", color='#2C3E50'),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='white',
            margin=dict(t=140, b=100, l=100, r=100)  # Increased margins
        )
        
        # Update axes with enhanced styling and better spacing
        fig.update_xaxes(
            title_font=dict(size=14, color='#34495E', family='Arial Black'),
            tickfont=dict(size=11, color='#2C3E50'),
            gridcolor='#E8E8E8',
            gridwidth=1,
            showline=True,
            linecolor='#BDC3C7',
            linewidth=2,
            zeroline=False,
            tickangle=45,
            tickmode='linear',
            automargin=True  # Prevent label cutoff
        )
        
        fig.update_yaxes(
            title_font=dict(size=14, color='#34495E', family='Arial Black'),
            tickfont=dict(size=11, color='#2C3E50'),
            gridcolor='#E8E8E8',
            gridwidth=1,
            showline=True,
            linecolor='#BDC3C7',
            linewidth=2,
            zeroline=False,
            automargin=True  # Prevent label cutoff
        )
        
        # Specific axis improvements for readability
        fig.update_xaxes(title_text="<b>Analysis Length (Characters)</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>AI Models</b>", row=1, col=1)
        fig.update_xaxes(title_text="<b>Response Time (seconds)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Analysis Length (characters)</b>", row=1, col=2)
        fig.update_xaxes(title_text="<b>Provider Type</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Average Cost (USD)</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Analysis Length (characters)</b>", row=2, col=2)
        fig.update_xaxes(title_text="<b>AI Models</b>", row=3, col=1)
        fig.update_yaxes(title_text="<b>Efficiency (chars/second)</b>", row=3, col=1)
        fig.update_xaxes(title_text="<b>Category Winners</b>", row=4, col=1)
        fig.update_yaxes(title_text="<b>Analysis Length (characters)</b>", row=4, col=1)
        
        # Improve tick spacing to prevent overlap
        fig.update_xaxes(tickangle=45, row=3, col=1)  # Rotate efficiency chart labels
        fig.update_xaxes(tickangle=0, row=4, col=1)   # Keep category winners straight
        
        # Save dashboard with explicit configuration
        dashboard_file = "ultimate_model_comparison.html"
        
        try:
            fig.write_html(
                dashboard_file,
                include_plotlyjs=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'ultimate_model_comparison',
                        'height': 1800,
                        'width': 1200,
                        'scale': 2
                    }
                },
                div_id="ultimate-model-comparison"
            )
            print(f"✅ Ultimate dashboard created: {dashboard_file}")
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            return None
        
        return dashboard_file
    
    def open_ultimate_dashboard(self):
        """Create and open the ultimate dashboard"""
        print("🌟 CREATING ULTIMATE MODEL COMPARISON")
        print("=" * 45)
        
        dashboard_file = self.create_ultimate_dashboard()
        
        if dashboard_file:
            # Open dashboard
            try:
                abs_path = os.path.abspath(dashboard_file)
                webbrowser.open(f'file://{abs_path}')
                print(f"🚀 Opened: {dashboard_file}")
            except Exception as e:
                print(f"❌ Failed to open: {e}")
            
            # Print comprehensive summary
            gemini_count = len([r for r in self.all_results if r['provider'] == 'Gemini'])
            ollama_count = len([r for r in self.all_results if r['provider'] == 'Ollama'])
            
            best_overall = max(self.all_results, key=lambda x: x['analysis_length'])
            best_local = max([r for r in self.all_results if r['provider'] == 'Ollama'], 
                           key=lambda x: x['analysis_length'])
            fastest = min(self.all_results, key=lambda x: x['response_time'])
            most_efficient = max(self.all_results, key=lambda x: x['efficiency'])
            
            print(f"\n🏆 ULTIMATE COMPARISON RESULTS:")
            print("=" * 40)
            print(f"📊 Total Models Compared: {len(self.all_results)}")
            print(f"   • Gemini (Cloud): {gemini_count} models")
            print(f"   • Ollama (Local): {ollama_count} models")
            print()
            print(f"🥇 Champions:")
            print(f"   🏆 Best Overall: {best_overall['model']} ({best_overall['analysis_length']:,} chars)")
            print(f"   💰 Best Local: {best_local['model']} ({best_local['analysis_length']:,} chars)")
            print(f"   ⚡ Fastest: {fastest['model']} ({fastest['response_time']:.2f}s)")
            print(f"   🚀 Most Efficient: {most_efficient['model']} ({most_efficient['efficiency']:.0f} chars/s)")
            print()
            print(f"💡 Key Insights:")
            local_percentage = (best_local['analysis_length'] / best_overall['analysis_length']) * 100
            print(f"   • Local models achieve {local_percentage:.1f}% of cloud quality")
            print(f"   • Local models are 100% FREE vs cloud costs")
            print(f"   • You have {ollama_count} different local options to choose from")
        
        return dashboard_file

def main():
    """Main function"""
    comparison = UltimateModelComparison()
    return comparison.open_ultimate_dashboard()

if __name__ == "__main__":
    main()