#!/usr/bin/env python3
"""
Gemini vs Local LLM Comparison Dashboard
Compares Gemini models with local Ollama models for fraud detection
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime

class GeminiVsLocalComparison:
    """Create comparison visualizations between Gemini and local models"""
    
    def __init__(self):
        """Initialize the comparison dashboard"""
        self.load_all_data()
        
    def load_all_data(self):
        """Load all analysis data"""
        # Load Gemini results
        try:
            with open('../data/comprehensive_gemini_analysis.json', 'r') as f:
                self.gemini_data = json.load(f)
            print("✅ Loaded Gemini analysis data")
        except FileNotFoundError:
            print("❌ Gemini analysis file not found")
            self.gemini_data = None
        
        # Load Ollama results
        try:
            with open('ollama_fraud_analysis_results.json', 'r') as f:
                self.ollama_data = json.load(f)
            print("✅ Loaded Ollama analysis data")
        except FileNotFoundError:
            print("❌ Ollama analysis file not found")
            self.ollama_data = None
        
        # Load alternative LLM results
        try:
            with open('alternative_llm_results.json', 'r') as f:
                self.alternative_data = json.load(f)
            print("✅ Loaded alternative LLM data")
        except FileNotFoundError:
            print("❌ Alternative LLM file not found")
            self.alternative_data = None
    
    def create_comparison_dashboard(self):
        """Create comprehensive comparison dashboard"""
        print("🆚 Creating Gemini vs Local LLM comparison dashboard...")
        
        # Create main comparison figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '<b>📊 Analysis Length Comparison</b>', '<b>⚡ Response Time Comparison</b>',
                '<b>💰 Cost Analysis (Cloud vs Local)</b>', '<b>🎯 Efficiency Analysis</b>',
                '<b>🏆 Top Performers</b>', '<b>📈 Performance Distribution</b>'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Prepare comparison data
        comparison_data = self.prepare_comparison_data()
        
        if not comparison_data:
            print("❌ No data available for comparison")
            return None
        
        # 1. Analysis Length Comparison
        models = [d['model'] for d in comparison_data]
        lengths = [d['analysis_length'] for d in comparison_data]
        providers = [d['provider'] for d in comparison_data]
        
        # Color by provider
        colors = []
        for provider in providers:
            if 'gemini' in provider.lower():
                colors.append('#4285F4')  # Google Blue
            elif 'ollama' in provider.lower():
                colors.append('#FF6B35')  # Orange for local
            else:
                colors.append('#34A853')  # Green for others
        
        fig.add_trace(go.Bar(
            x=models,
            y=lengths,
            name="Analysis Length",
            marker=dict(color=colors, line=dict(color='white', width=1)),
            text=[f'{l:,}' for l in lengths],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Length: %{y:,} chars<br>Provider: %{customdata}<extra></extra>",
            customdata=providers
        ), row=1, col=1)
        
        # 2. Response Time Comparison
        times = [d['response_time'] for d in comparison_data]
        
        fig.add_trace(go.Bar(
            x=models,
            y=times,
            name="Response Time",
            marker=dict(color=colors, line=dict(color='white', width=1)),
            text=[f'{t:.1f}s' for t in times],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Time: %{y:.2f}s<br>Provider: %{customdata}<extra></extra>",
            customdata=providers
        ), row=1, col=2)
        
        # 3. Cost Analysis
        costs = []
        cost_labels = []
        for d in comparison_data:
            if 'ollama' in d['provider'].lower():
                costs.append(0.0)  # Local is free
                cost_labels.append('FREE')
            else:
                # Estimate cost for cloud models (rough approximation)
                estimated_cost = (d['analysis_length'] / 1000) * 0.002  # ~$2 per 1M chars
                costs.append(estimated_cost)
                cost_labels.append(f'${estimated_cost:.4f}')
        
        fig.add_trace(go.Bar(
            x=models,
            y=costs,
            name="Estimated Cost",
            marker=dict(color=colors, line=dict(color='white', width=1)),
            text=cost_labels,
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Cost: %{text}<br>Provider: %{customdata}<extra></extra>",
            customdata=providers
        ), row=2, col=1)
        
        # 4. Efficiency Analysis (chars per second)
        efficiencies = [d['analysis_length'] / d['response_time'] for d in comparison_data]
        
        fig.add_trace(go.Scatter(
            x=times,
            y=lengths,
            mode='markers+text',
            text=[m.split(':')[0] if ':' in m else m.split('-')[-1] for m in models],
            textposition="top center",
            marker=dict(
                size=[max(10, min(30, eff/50)) for eff in efficiencies],
                color=colors,
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            name='Efficiency',
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Length: %{y:,} chars<br>Efficiency: %{customdata:.0f} chars/s<extra></extra>",
            customdata=efficiencies
        ), row=2, col=2)
        
        # 5. Top Performers
        top_by_length = sorted(comparison_data, key=lambda x: x['analysis_length'], reverse=True)[:8]
        top_models = [d['model'] for d in top_by_length]
        top_lengths = [d['analysis_length'] for d in top_by_length]
        top_providers = [d['provider'] for d in top_by_length]
        
        # Medal colors for top performers
        medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['#4285F4'] * 5
        
        fig.add_trace(go.Bar(
            x=top_models,
            y=top_lengths,
            name="Top Models",
            marker=dict(color=medal_colors, line=dict(color='white', width=2)),
            text=[f'{l:,}' for l in top_lengths],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Length: %{y:,} chars<br>Provider: %{customdata}<extra></extra>",
            customdata=top_providers
        ), row=3, col=1)
        
        # 6. Performance Distribution by Provider
        providers_unique = list(set(providers))
        for provider in providers_unique:
            provider_data = [d for d in comparison_data if d['provider'] == provider]
            provider_lengths = [d['analysis_length'] for d in provider_data]
            
            provider_color = '#4285F4' if 'gemini' in provider.lower() else '#FF6B35' if 'ollama' in provider.lower() else '#34A853'
            
            fig.add_trace(go.Box(
                y=provider_lengths,
                name=provider.title(),
                marker_color=provider_color,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>🆚 Gemini vs Local LLM Fraud Detection Comparison</b><br>" +
                     "<sub style='color: #7F8C8D;'>Cloud Models vs Local Ollama Models • Performance Analysis</sub>",
                x=0.5,
                font=dict(size=24, color='#2C3E50')
            ),
            height=1400,
            showlegend=False,
            template="plotly_white",
            font=dict(size=12, family="Arial, sans-serif"),
            paper_bgcolor='#FAFAFA',
            plot_bgcolor='white',
            margin=dict(t=120, b=80, l=80, r=80)
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="<b>Characters</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>Seconds</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>USD ($)</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>Response Time (s)</b>", row=2, col=2)
        fig.update_yaxes(title_text="<b>Analysis Length</b>", row=2, col=2)
        fig.update_yaxes(title_text="<b>Characters</b>", row=3, col=1)
        fig.update_yaxes(title_text="<b>Analysis Length</b>", row=3, col=2)
        
        # Save dashboard
        dashboard_file = "gemini_vs_local_comparison.html"
        fig.write_html(dashboard_file)
        print(f"✅ Comparison dashboard created: {dashboard_file}")
        
        return dashboard_file
    
    def prepare_comparison_data(self):
        """Prepare data for comparison"""
        comparison_data = []
        
        # Add Gemini data
        if self.gemini_data:
            gemini_results = self.gemini_data.get('analysis_results', [])
            successful_gemini = [r for r in gemini_results if r['status'] == 'success']
            
            for result in successful_gemini[:10]:  # Top 10 Gemini models
                comparison_data.append({
                    'model': result['model'].split('/')[-1].replace('gemini-', ''),
                    'provider': 'Gemini (Cloud)',
                    'analysis_length': result['analysis_length'],
                    'response_time': result['response_time']
                })
        
        # Add Ollama data
        if self.ollama_data:
            ollama_results = self.ollama_data.get('results', [])
            for result in ollama_results:
                comparison_data.append({
                    'model': result['model'],
                    'provider': 'Ollama (Local)',
                    'analysis_length': result['analysis_length'],
                    'response_time': result['response_time']
                })
        
        # Add alternative LLM data
        if self.alternative_data:
            alt_results = self.alternative_data.get('results', [])
            for result in alt_results:
                comparison_data.append({
                    'model': result['model'],
                    'provider': f"{result['provider']} (Local)",
                    'analysis_length': result['analysis_length'],
                    'response_time': result['response_time']
                })
        
        return comparison_data
    
    def create_detailed_analysis_comparison(self):
        """Create detailed analysis content comparison"""
        print("📝 Creating detailed analysis comparison...")
        
        # Get best performers from each category
        comparison_data = self.prepare_comparison_data()
        
        if not comparison_data:
            return None
        
        # Find best from each provider
        best_gemini = max([d for d in comparison_data if 'gemini' in d['provider'].lower()], 
                         key=lambda x: x['analysis_length'], default=None)
        best_local = max([d for d in comparison_data if 'local' in d['provider'].lower()], 
                        key=lambda x: x['analysis_length'], default=None)
        
        # Create comparison table
        fig = go.Figure()
        
        comparison_table = []
        if best_gemini:
            comparison_table.append([
                '🏆 Best Cloud (Gemini)',
                best_gemini['model'],
                f"{best_gemini['analysis_length']:,} chars",
                f"{best_gemini['response_time']:.2f}s",
                f"~${(best_gemini['analysis_length']/1000)*0.002:.4f}",
                '☁️ Cloud-based, High quality'
            ])
        
        if best_local:
            comparison_table.append([
                '🥇 Best Local (Ollama)',
                best_local['model'],
                f"{best_local['analysis_length']:,} chars",
                f"{best_local['response_time']:.2f}s",
                'FREE',
                '💻 Local, Private, No cost'
            ])
        
        # Add summary row
        if best_gemini and best_local:
            length_diff = best_gemini['analysis_length'] - best_local['analysis_length']
            time_diff = best_gemini['response_time'] - best_local['response_time']
            
            comparison_table.append([
                '📊 Difference',
                'Cloud vs Local',
                f"{length_diff:+,} chars",
                f"{time_diff:+.2f}s",
                f"${(best_gemini['analysis_length']/1000)*0.002:.4f} vs FREE",
                'Quality vs Cost tradeoff'
            ])
        
        fig.add_trace(go.Table(
            header=dict(
                values=['🎯 Category', '🤖 Model', '📊 Analysis Length', '⚡ Speed', '💰 Cost', '💡 Notes'],
                fill_color='#2C3E50',
                align='center',
                font=dict(size=14, color='white', family='Arial Black'),
                height=50
            ),
            cells=dict(
                values=list(zip(*comparison_table)) if comparison_table else [[], [], [], [], [], []],
                fill_color=[['#ECF0F1', '#FFFFFF'] * 2],
                align=['center', 'left', 'center', 'center', 'center', 'left'],
                font=dict(size=12, color='#2C3E50'),
                height=60
            )
        ))
        
        fig.update_layout(
            title=dict(
                text="<b>📊 Detailed Model Performance Comparison</b><br>" +
                     "<sub style='color: #7F8C8D;'>Cloud vs Local LLM Analysis</sub>",
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            height=400,
            template="plotly_white",
            paper_bgcolor='#FAFAFA',
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        comparison_file = "detailed_model_comparison_table.html"
        fig.write_html(comparison_file)
        print(f"✅ Detailed comparison created: {comparison_file}")
        
        return comparison_file
    
    def open_all_comparisons(self):
        """Open all comparison dashboards"""
        print("🚀 CREATING GEMINI VS LOCAL LLM COMPARISONS")
        print("=" * 55)
        
        files_created = []
        
        # Create main comparison dashboard
        main_file = self.create_comparison_dashboard()
        if main_file:
            files_created.append(main_file)
        
        # Create detailed comparison table
        detail_file = self.create_detailed_analysis_comparison()
        if detail_file:
            files_created.append(detail_file)
        
        # Open all files
        if files_created:
            print(f"\n🌐 Opening {len(files_created)} comparison dashboards...")
            
            for file in files_created:
                try:
                    abs_path = os.path.abspath(file)
                    webbrowser.open(f'file://{abs_path}')
                    print(f"🚀 Opened: {file}")
                except Exception as e:
                    print(f"❌ Failed to open {file}: {e}")
            
            print(f"\n🎉 COMPARISON COMPLETE!")
            print("=" * 35)
            print("📊 Dashboards Created:")
            for i, file in enumerate(files_created, 1):
                print(f"  {i}. {file}")
            
            # Show key insights
            comparison_data = self.prepare_comparison_data()
            if comparison_data:
                best_overall = max(comparison_data, key=lambda x: x['analysis_length'])
                fastest = min(comparison_data, key=lambda x: x['response_time'])
                
                print(f"\n🏆 KEY INSIGHTS:")
                print(f"   • Best Analysis: {best_overall['model']} ({best_overall['analysis_length']:,} chars)")
                print(f"   • Fastest Model: {fastest['model']} ({fastest['response_time']:.2f}s)")
                print(f"   • Total Models Compared: {len(comparison_data)}")
                print(f"   • Local vs Cloud: Mix of performance and cost benefits")
        
        return files_created

def main():
    """Main function"""
    comparison = GeminiVsLocalComparison()
    return comparison.open_all_comparisons()

if __name__ == "__main__":
    main()