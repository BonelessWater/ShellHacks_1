#!/usr/bin/env python3
"""
HTML Dashboard Diagnostic Tool
Check if HTML files are rendering properly
"""

import os
import webbrowser
from bs4 import BeautifulSoup
import json

def diagnose_html_file(filename):
    """Diagnose HTML file for common issues"""
    print(f"🔍 DIAGNOSING: {filename}")
    print("=" * 50)
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    # Check file size
    file_size = os.path.getsize(filename)
    print(f"📁 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    if file_size < 1000:
        print("⚠️  File seems too small - might be empty or corrupted")
        return False
    
    # Read and parse HTML
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 Content length: {len(content):,} characters")
        
        # Check for Plotly elements
        if 'Plotly.newPlot' in content:
            print("✅ Contains Plotly.newPlot - should render charts")
        else:
            print("❌ Missing Plotly.newPlot - charts won't render")
        
        if 'plotly-graph-div' in content:
            print("✅ Contains plotly-graph-div - has chart containers")
        else:
            print("❌ Missing plotly-graph-div - no chart containers")
        
        # Check for JavaScript errors
        if 'error' in content.lower() or 'exception' in content.lower():
            print("⚠️  Possible JavaScript errors found in content")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Check basic HTML structure
        if soup.find('html'):
            print("✅ Valid HTML structure")
        else:
            print("❌ Invalid HTML structure")
        
        # Check for div containers
        divs = soup.find_all('div', class_='plotly-graph-div')
        print(f"📊 Found {len(divs)} plotly graph containers")
        
        # Check for script tags
        scripts = soup.find_all('script')
        print(f"📜 Found {len(scripts)} script tags")
        
        # Look for data in scripts
        plotly_data_found = False
        for script in scripts:
            if script.string and 'Plotly.newPlot' in script.string:
                plotly_data_found = True
                # Extract a sample of the data
                script_content = script.string
                if '"data":[' in script_content:
                    print("✅ Found chart data in scripts")
                break
        
        if not plotly_data_found:
            print("❌ No Plotly chart data found in scripts")
        
        print(f"\n✅ DIAGNOSIS COMPLETE")
        print(f"   File appears to be {'VALID' if plotly_data_found and len(divs) > 0 else 'PROBLEMATIC'}")
        
        return plotly_data_found and len(divs) > 0
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_dashboard_files():
    """Test all dashboard HTML files"""
    print("🧪 TESTING ALL DASHBOARD FILES")
    print("=" * 60)
    
    html_files = [
        'ultimate_model_comparison.html',
        'fixed_ultimate_comparison.html',
        'professional_model_dashboard.html',
        'enterprise_model_dashboard.html',
        'professional_gemini_dashboard.html'
    ]
    
    results = {}
    
    for filename in html_files:
        print(f"\n{'='*20} {filename} {'='*20}")
        results[filename] = diagnose_html_file(filename)
        
        if results[filename]:
            print(f"✅ {filename} - WORKING")
        else:
            print(f"❌ {filename} - NEEDS FIXING")
            
        print()
    
    # Summary
    print(f"\n🎯 SUMMARY REPORT")
    print("=" * 40)
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"Working dashboards: {working_count}/{total_count}")
    
    for filename, working in results.items():
        status = "✅ WORKING" if working else "❌ BROKEN"
        print(f"  • {filename}: {status}")
    
    if working_count == total_count:
        print(f"\n🎉 ALL DASHBOARDS ARE WORKING!")
    else:
        print(f"\n⚠️  {total_count - working_count} dashboard(s) need attention")
    
    return results

if __name__ == "__main__":
    test_dashboard_files()