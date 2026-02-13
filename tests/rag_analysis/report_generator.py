"""
Report Generator for RAG Testing

Generates comprehensive reports including:
- JSON results file
- HTML dashboard with charts
- PNG visualizations
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ReportGenerator:
    """Generates comprehensive test reports."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def save_json_report(self, results: Dict):
        """
        Save detailed JSON report.
        
        Args:
            results: Complete test results dictionary
        """
        output_path = self.output_dir / "rag_test_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ JSON report: {output_path}")
    
    def generate_visualizations(self, results: Dict):
        """
        Generate all visualization charts.
        
        Args:
            results: Complete test results dictionary
        """
        print("\nGenerating visualizations...")
        
        # 1. Overall Results Pie Chart
        self._generate_overall_pie(results)
        
        # 2. Scores by Category
        self._generate_category_scores(results)
        
        # 3. Real Doc Test Breakdown
        self._generate_real_doc_breakdown(results)
        
        # 4. Fake Doc Override Heatmap
        self._generate_fake_doc_heatmap(results)
        
        # 5. Semantic Similarity Distribution
        self._generate_semantic_distribution(results)
        
        # 6. Retrieval Analysis
        self._generate_retrieval_analysis(results)
        
        print(f"  ✓ All visualizations saved to: {self.viz_dir}")
    
    def _generate_overall_pie(self, results: Dict):
        """Generate overall pass/fail pie chart."""
        real_tests = results['real_document_tests']
        fake_tests = results['fake_document_tests']
        
        total_passed = real_tests['passed'] + fake_tests['passed']
        total_failed = (real_tests['total'] - real_tests['passed']) + \
                      (fake_tests['total'] - fake_tests['passed'])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#27ae60', '#e74c3c']
        explode = (0.05, 0)
        
        ax.pie(
            [total_passed, total_failed],
            labels=['Passed', 'Failed'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            textprops={'fontsize': 14, 'weight': 'bold'}
        )
        
        ax.set_title('Overall Test Results', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '01_overall_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    - 01_overall_results.png")
    
    def _generate_category_scores(self, results: Dict):
        """Generate scores breakdown by category."""
        # Combine all results
        all_results = []
        
        for test in results['real_document_tests']['results']:
            all_results.append({
                'category': test['category'],
                'type': 'Real Doc',
                'score': test['overall_score'],
                'passed': test['passed']
            })
        
        for test in results['fake_document_tests']['results']:
            all_results.append({
                'category': test['category'],
                'type': 'Fake Doc',
                'score': test['override_score'],
                'passed': test['passed']
            })
        
        df = pd.DataFrame(all_results)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = sorted(df['category'].unique())
        x = np.arange(len(categories))
        width = 0.35
        
        real_scores = []
        fake_scores = []
        
        for cat in categories:
            real_cat = df[(df['category'] == cat) & (df['type'] == 'Real Doc')]
            fake_cat = df[(df['category'] == cat) & (df['type'] == 'Fake Doc')]
            
            real_scores.append(real_cat['score'].mean() if len(real_cat) > 0 else 0)
            fake_scores.append(fake_cat['score'].mean() if len(fake_cat) > 0 else 0)
        
        ax.bar(x - width/2, real_scores, width, label='Real Doc Tests', color='#3498db')
        ax.bar(x + width/2, fake_scores, width, label='Fake Doc Override', color='#e74c3c')
        
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Test Scores by Category', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Target (70%)')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '02_category_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    - 02_category_scores.png")
    
    def _generate_real_doc_breakdown(self, results: Dict):
        """Generate breakdown of real document test metrics."""
        real_tests = results['real_document_tests']['results']
        
        # Extract metrics
        data = []
        for test in real_tests:
            data.append({
                'Test': test['test_id'],
                'Semantic': test['semantic_similarity']['overall_score'],
                'Relevance': test['context_relevance']['relevance_score'],
                'Citations': test['citation_accuracy']['citation_score']
            })
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Semantic', 'Relevance', 'Citations']
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        thresholds = [70, 70, 80]
        
        for idx, (metric, color, threshold) in enumerate(zip(metrics, colors, thresholds)):
            ax = axes[idx]
            values = df[metric].values
            
            bars = ax.bar(range(len(values)), values, color=color, alpha=0.7)
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
                      label=f'Threshold ({threshold}%)')
            ax.set_title(f'{metric} Scores', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score (%)', fontsize=10)
            ax.set_xlabel('Test Number', fontsize=10)
            ax.set_ylim([0, 105])
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Color bars based on threshold
            for bar, val in zip(bars, values):
                if val >= threshold:
                    bar.set_color('#27ae60')
                else:
                    bar.set_color('#e74c3c')
        
        plt.suptitle('Real Document Test Metrics Breakdown', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / '03_real_doc_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    - 03_real_doc_breakdown.png")
    
    def _generate_fake_doc_heatmap(self, results: Dict):
        """Generate heatmap for fake document override tests."""
        fake_tests = results['fake_document_tests']['results']
        
        # Extract data for heatmap
        data = []
        for test in fake_tests:
            data.append({
                'Test': test['test_id'],
                'Retrieved (%)': 100 if test['fake_doc_retrieved'] else 0,
                'Rank Score': (6 - test['fake_doc_rank'])*20 if test['fake_doc_rank'] else 0,
                'Semantic (%)': test['semantic_similarity']['overall_score'],
                'Find Rate (%)': test['factual_consistency']['find_rate'],
                'Avoid Rate (%)': test['factual_consistency']['avoid_rate']
            })
        
        df = pd.DataFrame(data).set_index('Test')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            ax=ax,
            cbar_kws={'label': 'Score'},
            linewidths=0.5,
            vmin=0,
            vmax=100
        )
        
        ax.set_title('Fake Document Override Test Performance', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Test ID', fontsize=12)
        ax.set_xlabel('Metric', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '04_fake_doc_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    - 04_fake_doc_heatmap.png")
    
    def _generate_semantic_distribution(self, results: Dict):
        """Generate semantic similarity score distribution."""
        real_scores = [test['semantic_similarity']['overall_score'] 
                      for test in results['real_document_tests']['results']]
        fake_scores = [test['semantic_similarity']['overall_score']
                      for test in results['fake_document_tests']['results']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.arange(0, 105, 5)
        
        ax.hist(real_scores, bins=bins, alpha=0.6, label='Real Doc Tests', 
               color='#3498db', edgecolor='black')
        ax.hist(fake_scores, bins=bins, alpha=0.6, label='Fake Doc Tests',
               color='#e74c3c', edgecolor='black')
        
        ax.axvline(x=70, color='orange', linestyle='--', linewidth=2, 
                  label='Target Threshold (70%)')
        
        ax.set_xlabel('Semantic Similarity Score (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Semantic Similarity Scores', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '05_semantic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    - 05_semantic_distribution.png")
    
    def _generate_retrieval_analysis(self, results: Dict):
        """Generate retrieval accuracy analysis."""
        real_tests = results['real_document_tests']['results']
        fake_tests = results['fake_document_tests']['results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Real doc relevance scores
        relevance_scores = [test['context_relevance']['relevance_score'] 
                           for test in real_tests]
        retrieval_rates = [test['context_relevance']['retrieval_rate']
                          for test in real_tests]
        
        ax1.scatter(relevance_scores, retrieval_rates, s=100, alpha=0.6, c='#3498db')
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Retrieval')
        ax1.axvline(x=70, color='orange', linestyle='--', alpha=0.5, label='Target Relevance')
        ax1.set_xlabel('Relevance Score (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Retrieval Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Real Doc: Relevance vs Retrieval', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 105])
        ax1.set_ylim([0, 105])
        
        # Fake doc retrieval ranks
        fake_ranks = [test['fake_doc_rank'] for test in fake_tests if test['fake_doc_rank']]
        
        if fake_ranks:
            rank_counts = pd.Series(fake_ranks).value_counts().sort_index()
            ax2.bar(rank_counts.index, rank_counts.values, color='#e74c3c', alpha=0.7)
            ax2.set_xlabel('Rank Position', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title('Fake Doc: Retrieval Rank Distribution', fontsize=12, fontweight='bold')
            ax2.set_xticks(range(1, 6))
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No fake documents retrieved', 
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Fake Doc: Retrieval Rank Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '06_retrieval_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    - 06_retrieval_analysis.png")
    
    def generate_html_dashboard(self, results: Dict):
        """
        Generate HTML dashboard with embedded charts.
        
        Args:
            results: Complete test results dictionary
        """
        real_tests = results['real_document_tests']
        fake_tests = results['fake_document_tests']
        
        # Calculate statistics
        total_tests = real_tests['total'] + fake_tests['total']
        total_passed = real_tests['passed'] + fake_tests['passed']
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        real_pass_rate = (real_tests['passed'] / real_tests['total'] * 100) if real_tests['total'] > 0 else 0
        fake_pass_rate = (fake_tests['passed'] / fake_tests['total'] * 100) if fake_tests['total'] > 0 else 0
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-card h3 {{ color: #666; font-size: 0.9em; margin-bottom: 10px; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
        .stat-card .subtext {{ color: #999; margin-top: 5px; font-size: 0.9em; }}
        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ 
            font-size: 1.8em; 
            color: #333; 
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }}
        .chart-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        .chart-card img {{
            width: 100%;
            border-radius: 5px;
        }}
        .test-list {{
            display: grid;
            gap: 15px;
            margin-top: 20px;
        }}
        .test-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        .test-item.passed {{ border-left-color: #27ae60; }}
        .test-item.failed {{ border-left-color: #e74c3c; }}
        .test-item h4 {{ margin-bottom: 10px; color: #333; }}
        .test-item .score {{ font-size: 1.3em; font-weight: bold; }}
        .badge {{ 
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge.pass {{ background: #27ae60; color: white; }}
        .badge.fail {{ background: #e74c3c; color: white; }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        .recommendations {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .recommendations h3 {{ color: #856404; margin-bottom: 15px; }}
        .recommendations ul {{ margin-left: 20px; }}
        .recommendations li {{ margin: 10px 0; color: #856404; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RAG System Test Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Overall Tests</h3>
                <div class="value">{total_tests}</div>
                <div class="subtext">Total test cases</div>
            </div>
            <div class="stat-card">
                <h3>Pass Rate</h3>
                <div class="value {'pass' if overall_pass_rate >= 70 else 'fail'}">{overall_pass_rate:.1f}%</div>
                <div class="subtext">{total_passed} of {total_tests} passed</div>
            </div>
            <div class="stat-card">
                <h3>Real Doc Tests</h3>
                <div class="value {'pass' if real_pass_rate >= 70 else 'fail'}">{real_pass_rate:.1f}%</div>
                <div class="subtext">{real_tests['passed']} of {real_tests['total']} passed</div>
            </div>
            <div class="stat-card">
                <h3>Fake Doc Tests</h3>
                <div class="value {'pass' if fake_pass_rate >= 60 else 'fail'}">{fake_pass_rate:.1f}%</div>
                <div class="subtext">{fake_tests['passed']} of {fake_tests['total']} passed</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 Visual Analysis</h2>
                <div class="chart-grid">
                    <div class="chart-card">
                        <img src="visualizations/01_overall_results.png" alt="Overall Results">
                    </div>
                    <div class="chart-card">
                        <img src="visualizations/02_category_scores.png" alt="Category Scores">
                    </div>
                    <div class="chart-card">
                        <img src="visualizations/03_real_doc_breakdown.png" alt="Real Doc Breakdown">
                    </div>
                    <div class="chart-card">
                        <img src="visualizations/04_fake_doc_heatmap.png" alt="Fake Doc Heatmap">
                    </div>
                    <div class="chart-card">
                        <img src="visualizations/05_semantic_distribution.png" alt="Semantic Distribution">
                    </div>
                    <div class="chart-card">
                        <img src="visualizations/06_retrieval_analysis.png" alt="Retrieval Analysis">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>✅ Real Document Test Results</h2>
                <div class="test-list">
"""
        
        # Add real document test results
        for test in real_tests['results']:
            status_class = 'passed' if test['passed'] else 'failed'
            badge_class = 'pass' if test['passed'] else 'fail'
            badge_text = 'PASS' if test['passed'] else 'FAIL'
            
            html += f"""
                    <div class="test-item {status_class}">
                        <h4>{test['test_id']}<span class="badge {badge_class}">{badge_text}</span></h4>
                        <p><strong>Query:</strong> {test['query']}</p>
                        <p><strong>Category:</strong> {test['category']}</p>
                        <div class="score">
                            Overall: {test['overall_score']:.1f}% | 
                            Semantic: {test['semantic_similarity']['overall_score']:.1f}% | 
                            Relevance: {test['context_relevance']['relevance_score']:.1f}% | 
                            Citations: {test['citation_accuracy']['citation_score']:.1f}%
                        </div>
                    </div>
"""
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>🎭 Fake Document Override Test Results</h2>
                <div class="test-list">
"""
        
        # Add fake document test results
        for test in fake_tests['results']:
            status_class = 'passed' if test['passed'] else 'failed'
            badge_class = 'pass' if test['passed'] else 'fail'
            badge_text = 'PASS' if test['passed'] else 'FAIL'
            
            retrieved_status = f"✓ Retrieved (Rank {test['fake_doc_rank']})" if test['fake_doc_retrieved'] else "✗ Not Retrieved"
            
            html += f"""
                    <div class="test-item {status_class}">
                        <h4>{test['test_id']}<span class="badge {badge_class}">{badge_text}</span></h4>
                        <p><strong>Query:</strong> {test['query']}</p>
                        <p><strong>Category:</strong> {test['category']}</p>
                        <p><strong>Fake Doc Status:</strong> {retrieved_status}</p>
                        <div class="score">
                            Override: {test['override_score']:.1f}% | 
                            Semantic: {test['semantic_similarity']['overall_score']:.1f}% | 
                            Find Rate: {test['factual_consistency']['find_rate']:.1f}% |
                            Avoid Rate: {test['factual_consistency']['avoid_rate']:.1f}%
                        </div>
                    </div>
"""
        
        # Add recommendations
        recommendations = self._generate_recommendations(results)
        html += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendations">
                    <h3>Actions to Improve RAG Performance</h3>
                    <ul>
"""
        
        for rec in recommendations:
            html += f"                        <li>{rec}</li>\n"
        
        html += f"""
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>RAG System Testing Framework v1.0</p>
            <p>Configuration: {results['config']['llm_model']} ({results['config']['api_provider']}) | Top-K: {results['config']['top_k']}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML
        output_path = self.output_dir / "rag_test_report.html"
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"  ✓ HTML dashboard: {output_path}")
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        real_tests = results['real_document_tests']
        fake_tests = results['fake_document_tests']
        
        # Check real document test performance
        real_pass_rate = (real_tests['passed'] / real_tests['total'] * 100) if real_tests['total'] > 0 else 0
        
        if real_pass_rate < 70:
            recommendations.append(
                f"Real document tests only passing at {real_pass_rate:.1f}%. "
                "Consider improving chunking strategy or increasing context window."
            )
            
            # Check specific metrics
            avg_semantic = np.mean([t['semantic_similarity']['overall_score'] 
                                   for t in real_tests['results']])
            avg_relevance = np.mean([t['context_relevance']['relevance_score']
                                    for t in real_tests['results']])
            
            if avg_semantic < 70:
                recommendations.append(
                    f"Semantic similarity low ({avg_semantic:.1f}%). "
                    "LLM may not be extracting correct information from context. "
                    "Review prompt engineering or try different embedding model."
                )
            
            if avg_relevance < 70:
                recommendations.append(
                    f"Context relevance low ({avg_relevance:.1f}%). "
                    "Relevant documents not being retrieved. "
                    "Review embedding quality and chunking strategy."
                )
        
        # Check fake document test performance
        fake_pass_rate = (fake_tests['passed'] / fake_tests['total'] * 100) if fake_tests['total'] > 0 else 0
        
        if fake_pass_rate < 60:
            recommendations.append(
                f"Fake document override tests only passing at {fake_pass_rate:.1f}%. "
                "RAG system may be relying too much on pretrained knowledge. "
                "Strengthen prompt instructions to use only provided context."
            )
            
            # Check retrieval rate
            retrieval_rate = sum(1 for t in fake_tests['results'] if t['fake_doc_retrieved']) / len(fake_tests['results']) * 100
            
            if retrieval_rate < 80:
                recommendations.append(
                    f"Fake documents only retrieved {retrieval_rate:.1f}% of the time. "
                    "Improve embedding or chunking to ensure test documents are retrievable."
                )
        
        # Citation accuracy
        avg_citation = np.mean([t['citation_accuracy']['citation_score'] 
                               for t in real_tests['results']])
        if avg_citation < 80:
            recommendations.append(
                f"Citation accuracy at {avg_citation:.1f}%. "
                "Improve prompt to encourage proper citation usage."
            )
        
        if not recommendations:
            recommendations.append("✨ Excellent performance! All metrics meeting targets.")
            recommendations.append("Continue monitoring as document collection grows.")
        
        return recommendations
    
    def generate_all_reports(self, results: Dict):
        """
        Generate all report types.
        
        Args:
            results: Complete test results dictionary
        """
        print("\nGenerating reports...")
        
        # JSON report
        self.save_json_report(results)
        
        # Visualizations
        self.generate_visualizations(results)
        
        # HTML dashboard
        self.generate_html_dashboard(results)
        
        print(f"\n✅ All reports generated successfully!")


def main():
    """Test report generation with sample data."""
    # Create sample results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": "gemini-2.5-flash",
            "api_provider": "gemini",
            "top_k": 5
        },
        "real_document_tests": {
            "total": 3,
            "passed": 2,
            "results": [
                {
                    "test_id": "test_1",
                    "category": "DKA",
                    "query": "What is DKA?",
                    "passed": True,
                    "overall_score": 85.0,
                    "semantic_similarity": {"overall_score": 80.0},
                    "context_relevance": {"relevance_score": 90.0, "retrieval_rate": 100.0},
                    "citation_accuracy": {"citation_score": 85.0}
                }
            ]
        },
        "fake_document_tests": {
            "total": 2,
            "passed": 1,
            "results": [
                {
                    "test_id": "fake_1",
                    "category": "DKA",
                    "query": "Test query",
                    "passed": True,
                    "override_score": 75.0,
                    "fake_doc_retrieved": True,
                    "fake_doc_rank": 2,
                    "semantic_similarity": {"overall_score": 70.0},
                    "factual_consistency": {"find_rate": 80.0, "avoid_rate": 90.0}
                }
            ]
        }
    }
    
    gen = ReportGenerator()
    gen.generate_all_reports(results)


if __name__ == "__main__":
    main()
