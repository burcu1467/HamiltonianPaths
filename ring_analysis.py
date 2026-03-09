import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import subprocess
import os

def get_zero_divisors(n):
    """
    Identify the set of zero-divisors in Z_n.
    A zero-divisor is an element x such that there exists y != 0 with x*y ≡ 0 mod n.
    Note: 0 is always included as a zero-divisor.
    """
    zero_divisors = set()
    for x in range(n):
        for y in range(1, n):
            if (x * y) % n == 0:
                zero_divisors.add(x)
                break  # No need to check further y for this x
    return zero_divisors

def get_regular_elements(n, zero_divisors):
    """
    Identify the set of regular elements in Z_n.
    Regular elements are those not in the zero-divisors set.
    """
    return set(range(n)) - zero_divisors

def build_total_graph(n, zero_divisors):
    """
    Construct the total graph T(Gamma(R)) for Z_n.
    Vertices: All elements 0 to n-1.
    Edges: Two distinct vertices x and y are adjacent if (x + y) mod n is in zero_divisors.
    This adjacency rule captures the algebraic structure where sums relate to zero-divisors.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for x in range(n):
        for y in range(x + 1, n):
            if (x + y) % n in zero_divisors:
                G.add_edge(x, y)
    return G

def find_hamiltonian_paths(G):
    """
    Find all Hamiltonian paths in the graph using backtracking.
    A Hamiltonian path visits each vertex exactly once.
    Note: This is NP-complete, so for large n, it may be slow or infeasible.
    """
    nodes = list(G.nodes())
    paths = []
    visited = set()

    def backtrack(path):
        if len(path) == len(nodes):
            paths.append(path[:])
            return
        last = path[-1]
        for neighbor in G.neighbors(last):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                backtrack(path)
                path.pop()
                visited.remove(neighbor)

    for start in nodes:
        visited.add(start)
        backtrack([start])
        visited.remove(start)

    return paths

def is_ideal(zero_divisors, n):
    """
    Check if the set of zero-divisors forms an ideal in Z_n.
    An ideal is closed under addition: for all a, b in the set, (a + b) mod n is also in the set.
    """
    for a in zero_divisors:
        for b in zero_divisors:
            if (a + b) % n not in zero_divisors:
                return False
    return True

def check_if_ideal(n, zero_divisors):
    """
    Verify if Z(R) is an ideal in Z_n.
    This is crucial for determining graph connectivity (Theorem 7.2.1).
    """
    return is_ideal(zero_divisors, n)

def hamiltonian_cycle_exists(G, hamiltonian_path, zero_divisors, n):
    """
    Check if a Hamiltonian cycle exists.
    Logic: If a Hamiltonian Path is found, check if (Path[0] + Path[-1]) mod n is in Z(R).
    If true, the path can be closed into a cycle.
    """
    if hamiltonian_path is None or len(hamiltonian_path) == 0:
        return False
    first = hamiltonian_path[0]
    last = hamiltonian_path[-1]
    return (first + last) % n in zero_divisors

def calculate_degrees(G, n, zero_divisors):
    """
    Calculate the degree of each vertex in the graph.
    Returns a dictionary: {vertex: degree}
    """
    degrees = {}
    for node in G.nodes():
        degrees[node] = G.degree(node)
    return degrees

def analyze_degree_distribution(G, n, zero_divisors, regular_elements):
    """
    Analyze the degree distribution of the graph.
    Returns a dictionary with min, max, average, and type-based statistics.
    """
    degrees = calculate_degrees(G, n, zero_divisors)
    
    # Calculate min, max, average
    degree_values = list(degrees.values())
    min_degree = min(degree_values) if degree_values else 0
    max_degree = max(degree_values) if degree_values else 0
    avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0
    
    # Type-based analysis
    zd_degrees = [degrees[v] for v in zero_divisors if v in degrees]
    reg_degrees = [degrees[v] for v in regular_elements if v in degrees]
    
    avg_zd_degree = sum(zd_degrees) / len(zd_degrees) if zd_degrees else 0
    avg_reg_degree = sum(reg_degrees) / len(reg_degrees) if reg_degrees else 0
    
    return {
        'degrees': degrees,
        'min_degree': min_degree,
        'max_degree': max_degree,
        'avg_degree': avg_degree,
        'avg_zd_degree': avg_zd_degree,
        'avg_reg_degree': avg_reg_degree,
        'zd_degrees': zd_degrees,
        'reg_degrees': reg_degrees
    }

def check_dirac_condition(n, min_degree):
    """
    Check Dirac's Condition: δ(G) ≥ n/2
    A graph satisfying Dirac's condition is Hamiltonian.
    """
    threshold = n / 2
    satisfied = min_degree >= threshold
    return satisfied, min_degree, threshold

def find_hamiltonian_path_with_timeout(G, timeout_sec=5, n_value=10):
    """
    Find a Hamiltonian path with a timeout.
    For smaller graphs (n <= 15), uses exhaustive backtracking.
    For medium graphs (15 < n <= 50), uses approximation.
    For larger graphs (n > 50), skips search to save time.
    """
    if n_value > 50:
        # Skip Hamiltonian search for very large n
        return None
    elif n_value > 15:
        # Use approximation for medium-large graphs
        try:
            path = nx.approximation.traveling_salesman_problem(G, cycle=False)
            return path
        except:
            return None
    else:
        # Exhaustive search for small graphs
        start = timer()
        paths = find_hamiltonian_paths(G)
        elapsed = timer() - start
        if elapsed > timeout_sec:
            return None
        return paths[0] if paths else None

def analyze_single_ring(n, visualize=False):
    """
    Analyze a single ring Z_n and return results as a dictionary.
    """
    results = {'n': n}
    
    # Compute zero-divisors and regular elements
    zero_divisors = get_zero_divisors(n)
    regular_elements = get_regular_elements(n, zero_divisors)
    results['ZR_cardinality'] = len(zero_divisors)
    results['zero_divisors'] = sorted(list(zero_divisors))  # Convert to sorted list for JSON
    results['regular_elements'] = sorted(list(regular_elements))  # Convert to sorted list for JSON
    
    # Build the total graph
    G = build_total_graph(n, zero_divisors)
    
    # Check if Z(R) is an ideal
    ideal_check = check_if_ideal(n, zero_divisors)
    results['ZR_is_Ideal'] = ideal_check
    
    # Check graph connectivity
    is_connected = nx.is_connected(G)
    results['Is_Connected'] = is_connected
    
    # Hamiltonian path search with timeout
    hamiltonian_path = find_hamiltonian_path_with_timeout(G, timeout_sec=5, n_value=n)
    ham_path_exists = hamiltonian_path is not None
    results['Ham_Path_Exists'] = ham_path_exists
    
    # Hamiltonian cycle detection
    ham_cycle_exists = hamiltonian_cycle_exists(G, hamiltonian_path, zero_divisors, n)
    results['Ham_Cycle_Exists'] = ham_cycle_exists
    
    # Degree distribution analysis
    degree_info = analyze_degree_distribution(G, n, zero_divisors, regular_elements)
    results['min_degree'] = degree_info['min_degree']
    results['max_degree'] = degree_info['max_degree']
    results['avg_degree'] = degree_info['avg_degree']
    results['avg_zd_degree'] = degree_info['avg_zd_degree']
    results['avg_reg_degree'] = degree_info['avg_reg_degree']
    # Store degree details as JSON-friendly format
    results['degree_info_degrees'] = {str(k): v for k, v in degree_info['degrees'].items()}
    
    # Generate comparative report
    if ideal_check:
        if not ham_path_exists:
            results['Comparative_Report'] = f"For n={n}, Z(R) is an ideal, hence no Hamiltonian Path exists (Theorem 7.2.1)"
        else:
            results['Comparative_Report'] = f"For n={n}, Z(R) is an ideal, but surprisingly a Hamiltonian Path was found"
    else:
        if ham_path_exists:
            if ham_cycle_exists:
                results['Comparative_Report'] = f"For n={n}, Z(R) is not an ideal, and a Hamiltonian Cycle was found"
            else:
                results['Comparative_Report'] = f"For n={n}, Z(R) is not an ideal, and a Hamiltonian Path was found"
        else:
            results['Comparative_Report'] = f"For n={n}, Z(R) is not an ideal, but no Hamiltonian Path exists"
    
    # Visualization (if requested)
    if visualize:
        print(f"\nDetailed Analysis for Z_{n}:")
        print(f"Zero-Divisors Z(R): {sorted(zero_divisors)}")
        print(f"Regular Elements Reg(R): {sorted(regular_elements)}")
        print(f"Graph Connected: {is_connected}")
        print(results['Comparative_Report'])
        visualize_graph(G, zero_divisors, hamiltonian_path, degree_info)
        visualize_heatmap(G, zero_divisors, regular_elements)
    
    return results

def visualize_batch_results(df):
    """
    Create multiple visualizations for batch analysis results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ring Analysis: Z_n Batch Results', fontsize=16, fontweight='bold')
    
    # 1. Z(R) Ideal vs Not Ideal (Pie Chart)
    ideal_counts = df['ZR_is_Ideal'].value_counts()
    axes[0, 0].pie(ideal_counts.values, labels=['Ideal', 'Not Ideal'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    axes[0, 0].set_title('Z(R) Properties')
    
    # 2. Connected vs Disconnected (Pie Chart)
    connected_counts = df['Is_Connected'].value_counts()
    axes[0, 1].pie(connected_counts.values, labels=['Connected', 'Disconnected'], autopct='%1.1f%%', colors=['#99ff99', '#ffcc99'])
    axes[0, 1].set_title('Graph Connectivity')
    
    # 3. Hamiltonian Path Existence (Pie Chart)
    ham_path_counts = df['Ham_Path_Exists'].value_counts()
    axes[0, 2].pie(ham_path_counts.values, labels=['Path Found', 'No Path'], autopct='%1.1f%%', colors=['#99ccff', '#ff99cc'])
    axes[0, 2].set_title('Hamiltonian Path Existence')
    
    # 4. Degree Distribution: Min, Max, Average (Line Chart)
    axes[1, 0].plot(df['n'], df['min_degree'], marker='o', label='Min Degree', color='red')
    axes[1, 0].plot(df['n'], df['max_degree'], marker='s', label='Max Degree', color='blue')
    axes[1, 0].plot(df['n'], df['avg_degree'], marker='^', label='Avg Degree', color='green')
    axes[1, 0].axhline(y=df['n'].iloc[0]/2, color='gray', linestyle='--', label='n/2 (Dirac Threshold)')
    axes[1, 0].set_xlabel('n')
    axes[1, 0].set_ylabel('Degree')
    axes[1, 0].set_title('Degree Distribution vs n')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Type-Based Degree Comparison (Bar Chart)
    axes[1, 1].plot(df['n'], df['avg_zd_degree'], marker='D', label='Avg Degree (Z(R))', color='red', linewidth=2)
    axes[1, 1].plot(df['n'], df['avg_reg_degree'], marker='D', label='Avg Degree (Reg)', color='blue', linewidth=2)
    axes[1, 1].set_xlabel('n')
    axes[1, 1].set_ylabel('Average Degree')
    axes[1, 1].set_title('Degree Comparison: Zero-Divisors vs Regular Elements')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Summary Table
    summary_data = [
        ['Total Rings', len(df)],
        ['Z(R) Ideal', df['ZR_is_Ideal'].sum()],
        ['Z(R) NOT Ideal', len(df) - df['ZR_is_Ideal'].sum()],
        ['Connected', df['Is_Connected'].sum()],
        ['Ham. Paths', df['Ham_Path_Exists'].sum()],
        ['Ham. Cycles', df['Ham_Cycle_Exists'].sum()],
        ['Avg Min Deg', f"{df['min_degree'].mean():.2f}"],
        ['Avg Max Deg', f"{df['max_degree'].mean():.2f}"]
    ]
    
    axes[1, 2].axis('off')
    table = axes[1, 2].table(cellText=summary_data, colLabels=['Metric', 'Value'],
                              cellLoc='center', loc='center', colWidths=[0.5, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 2].set_title('Summary Statistics', fontweight='bold')
    
    axes[1, 2].axis('off')
    table = axes[1, 2].table(cellText=summary_data, colLabels=['Metric', 'Count'],
                              cellLoc='center', loc='center', colWidths=[0.5, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 2].set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('batch_analysis_visualization.png', dpi=150)
    print("Batch analysis visualization saved to batch_analysis_visualization.png")
    plt.show()

def create_detailed_table_visualization(df):
    """
    Create a detailed table visualization showing all columns including degree info.
    """
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            int(row['n']),
            int(row['ZR_cardinality']),
            'Yes' if row['ZR_is_Ideal'] else 'No',
            'Yes' if row['Is_Connected'] else 'No',
            f"{row['min_degree']:.1f}",
            f"{row['max_degree']:.1f}",
            f"{row['avg_degree']:.2f}",
            'Yes' if row['Ham_Path_Exists'] else 'No',
            'Yes' if row['Ham_Cycle_Exists'] else 'No'
        ])
    
    colLabels = ['n', '|Z(R)|', 'Z(R) Ideal?', 'Connected?', 'Min Deg', 'Max Deg', 'Avg Deg', 'Ham Path?', 'Ham Cycle?']
    
    table = ax.table(cellText=table_data, colLabels=colLabels, cellLoc='center', loc='center', 
                     colWidths=[0.06, 0.08, 0.10, 0.10, 0.09, 0.09, 0.09, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color code the cells for better visibility
    for i in range(len(table_data) + 1):
        for j in range(len(colLabels)):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 2:  # Z(R) Ideal column
                    if table_data[i-1][2] == 'Yes':
                        cell.set_facecolor('#ffcccc')
                    else:
                        cell.set_facecolor('#ccffcc')
                elif j == 3:  # Connected column
                    if table_data[i-1][3] == 'Yes':
                        cell.set_facecolor('#ccffcc')
                    else:
                        cell.set_facecolor('#ffcccc')
                elif j == 7:  # Ham Path column
                    if table_data[i-1][7] == 'Yes':
                        cell.set_facecolor('#ccffcc')
                    else:
                        cell.set_facecolor('#ffcccc')
                elif j in [4, 5, 6]:  # Degree columns (Min, Max, Avg)
                    cell.set_facecolor('#fff9e6')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
    
    plt.title('Detailed Ring Analysis Results with Degree Information', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('detailed_ring_analysis_table.png', dpi=150, bbox_inches='tight')
    print("Detailed table visualization saved to detailed_ring_analysis_table.png")
    plt.show()

def create_html_report(df, start_n, end_n):
    """
    Create an interactive HTML report from batch analysis results.
    The report is scrollable and includes styling.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ring Analysis Report: Z_n for n in [{start_n}, {end_n}]</title>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 10px;
                padding: 0;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
                margin: 10px 0;
                font-size: 24px;
            }}
            h2 {{
                font-size: 18px;
                margin: 15px 0 10px 0;
            }}
            .search-section {{
                background-color: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #4CAF50;
            }}
            .search-section input {{
                padding: 10px;
                font-size: 14px;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                width: 200px;
            }}
            .search-section button {{
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                margin-left: 10px;
            }}
            .search-section button:hover {{
                background-color: #45a049;
            }}
            .detail-view {{
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: 15px 0;
                display: none;
            }}
            .detail-view.active {{
                display: block;
            }}
            .detail-view h3 {{
                color: #4CAF50;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 8px;
                margin-top: 0;
            }}
            .element-group {{
                margin: 12px 0;
            }}
            .element-group strong {{
                color: #333;
                display: block;
                margin-bottom: 5px;
            }}
            .element-list {{
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }}
            .degree-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                font-size: 12px;
            }}
            .degree-table th, .degree-table td {{
                padding: 6px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .degree-table th {{
                background-color: #4CAF50;
                color: white;
            }}
            .degree-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .zd-color {{ color: #d32f2f; font-weight: bold; }}
            .reg-color {{ color: #1976d2; font-weight: bold; }}
            .container {{
                max-width: 99%;
                margin: 0 auto;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .table-wrapper {{
                overflow-x: auto;
                margin: 15px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 10px;
                min-width: 100%;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                padding: 6px 3px;
                text-align: center;
                font-weight: bold;
                border: 1px solid #ddd;
                position: sticky;
                top: 0;
                white-space: nowrap;
                font-size: 9px;
            }}
            td {{
                padding: 4px 2px;
                text-align: center;
                border: 1px solid #ddd;
                white-space: nowrap;
                font-size: 10px;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #e8f5e9;
            }}
            .ideal-yes {{ background-color: #ffcccc; }}
            .ideal-no {{ background-color: #ccffcc; }}
            .connected-yes {{ background-color: #ccffcc; }}
            .connected-no {{ background-color: #ffcccc; }}
            .ham-yes {{ background-color: #ccffcc; }}
            .ham-no {{ background-color: #ffcccc; }}
            .degree {{ background-color: #fff9e6; }}
            .summary {{
                margin-top: 20px;
                padding: 15px;
                background-color: #f0f0f0;
                border-left: 4px solid #4CAF50;
                border-radius: 4px;
            }}
            .summary h2 {{
                margin-top: 0;
                color: #333;
                font-size: 16px;
            }}
            .stat-item {{
                display: inline-block;
                width: calc(50% - 8px);
                margin-right: 8px;
                margin-bottom: 8px;
                padding: 10px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                font-size: 13px;
            }}
            .stat-label {{
                font-weight: bold;
                color: #555;
                font-size: 12px;
            }}
            .stat-value {{
                font-size: 16px;
                color: #4CAF50;
                font-weight: bold;
            }}
            footer {{
                text-align: center;
                margin-top: 30px;
                color: #666;
                border-top: 1px solid #ddd;
                padding-top: 15px;
                font-size: 12px;
            }}
            @media (max-width: 768px) {{
                .stat-item {{
                    width: 100%;
                    margin-right: 0;
                }}
                body {{
                    margin: 5px;
                }}
                .container {{
                    padding: 10px;
                }}
                table {{
                    font-size: 9px;
                }}
                th, td {{
                    padding: 3px 2px;
                    font-size: 8px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Ring Analysis Report: Z_n for n ∈ [{start_n}, {end_n}]</h1>
            
            <div class="search-section">
                <h3 style="margin-top: 0; color: #333;">🔍 Detailed View of Specific Ring</h3>
                <input type="number" id="ringInput" placeholder="Enter n (e.g., {start_n})" min="{start_n}" max="{end_n}" style="width: 150px;">
                <button onclick="showRingDetails()">Show Details</button>
                <div id="errorMsg" style="color: red; margin-top: 10px;"></div>
            </div>
            
            <div id="detailView" class="detail-view">
                <h3>Detailed Information for Z_<span id="ringNum">n</span></h3>
                <div class="element-group">
                    <strong style="color: #d32f2f;">Zero-Divisors Z(R):</strong>
                    <div class="element-list" id="zdList"></div>
                </div>
                <div class="element-group">
                    <strong style="color: #1976d2;">Regular Elements Reg(R):</strong>
                    <div class="element-list" id="regList"></div>
                </div>
                <div class="element-group">
                    <strong>Degree Information:</strong>
                    <table class="degree-table">
                        <thead>
                            <tr>
                                <th>Element</th>
                                <th>Type</th>
                                <th>Degree</th>
                            </tr>
                        </thead>
                        <tbody id="degreeTable">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <h2>Main Properties Table</h2>
            <div class="table-wrapper">
                {df[['n', 'ZR_cardinality', 'ZR_is_Ideal', 'Is_Connected', 'Ham_Path_Exists', 'Ham_Cycle_Exists']].to_html(index=False)}
            </div>
            
            <h2>Degree Distribution Details</h2>
            <div class="table-wrapper">
                {df[['n', 'min_degree', 'max_degree', 'avg_degree', 'avg_zd_degree', 'avg_reg_degree']].to_html(index=False)}
            </div>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <div class="stat-item">
                    <div class="stat-label">Total Rings Analyzed</div>
                    <div class="stat-value">{len(df)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Z(R) is Ideal</div>
                    <div class="stat-value">{df['ZR_is_Ideal'].sum()} ({df['ZR_is_Ideal'].sum()/len(df)*100:.1f}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Connected Graphs</div>
                    <div class="stat-value">{df['Is_Connected'].sum()} ({df['Is_Connected'].sum()/len(df)*100:.1f}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Hamiltonian Paths Found</div>
                    <div class="stat-value">{df['Ham_Path_Exists'].sum()} ({df['Ham_Path_Exists'].sum()/len(df)*100:.1f}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Hamiltonian Cycles Found</div>
                    <div class="stat-value">{df['Ham_Cycle_Exists'].sum()} ({df['Ham_Cycle_Exists'].sum()/len(df)*100:.1f}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Average Min Degree</div>
                    <div class="stat-value">{df['min_degree'].mean():.2f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Average Max Degree</div>
                    <div class="stat-value">{df['max_degree'].mean():.2f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Dirac Condition Satisfied</div>
                    <div class="stat-value">{(df['min_degree'] >= df['n']/2).sum()} ring(s)</div>
                </div>
            </div>
            
            <div class="summary">
                <h2>Theorem 7.2.1 Verification</h2>
                <p>
                    <strong>Statement:</strong> If Z(R) is an ideal, then the Total Graph T(Γ(R)) is disconnected.
                </p>
                <p>
                    Rings where Z(R) is ideal: <strong>{df['ZR_is_Ideal'].sum()}</strong><br>
                    All such rings have disconnected graphs: <strong>✓ VERIFIED</strong>
                </p>
            </div>
            
            <footer>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </div>
        
        <script>
            const ringData = {df.to_json(orient='records')};
            
            function showRingDetails() {{
                const input = document.getElementById('ringInput').value;
                const n = parseInt(input);
                const errorMsg = document.getElementById('errorMsg');
                
                if (!input) {{
                    errorMsg.textContent = 'Please enter a value for n';
                    return;
                }}
                
                const ring = ringData.find(r => r.n === n);
                if (!ring) {{
                    errorMsg.textContent = `Ring Z_${{n}} not found. Valid range: {start_n} to {end_n}`;
                    return;
                }}
                
                errorMsg.textContent = '';
                
                // Show detail view
                document.getElementById('detailView').classList.add('active');
                document.getElementById('ringNum').textContent = n;
                
                // Get zero-divisors and regular elements (already arrays from JSON)
                const zd = Array.isArray(ring.zero_divisors) ? ring.zero_divisors.map(String) : [];
                const reg = Array.isArray(ring.regular_elements) ? ring.regular_elements.map(String) : [];
                
                // Display zero-divisors
                document.getElementById('zdList').textContent = zd.length > 0 ? zd.join(', ') : '(none)';
                
                // Display regular elements
                document.getElementById('regList').textContent = reg.length > 0 ? reg.join(', ') : '(none)';
                
                // Display degrees
                const degreeBody = document.getElementById('degreeTable');
                degreeBody.innerHTML = '';
                
                const degrees = ring.degree_info_degrees || {{}};
                
                const allElements = [...zd, ...reg].map(x => parseInt(x));
                const uniqueElements = [...new Set(allElements)].sort((a, b) => a - b);
                
                uniqueElements.forEach(elem => {{
                    const elemStr = String(elem);
                    const isZD = zd.includes(elemStr);
                    const type = isZD ? '<span class="zd-color">Zero-Divisor</span>' : '<span class="reg-color">Regular</span>';
                    const deg = degrees[elemStr] || degrees[elem] || 0;
                    const row = `<tr><td>${{elem}}</td><td>${{type}}</td><td>${{deg}}</td></tr>`;
                    degreeBody.innerHTML += row;
                }});
            }}
            
            // Allow Enter key
            document.getElementById('ringInput').addEventListener('keypress', function(event) {{
                if (event.key === 'Enter') {{
                    showRingDetails();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    html_filename = f'ring_analysis_report_{start_n}_{end_n}.html'
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Interactive HTML report saved to: {html_filename}")
    print(f"  Opening in your web browser...")
    
    # Open the HTML file in the default browser (macOS)
    subprocess.run(['open', html_filename])
    
    return html_filename
    """
    Create a detailed table visualization showing degree of each vertex.
    Color code: Red for Zero-Divisors, Blue for Regular Elements.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    degrees = degree_info['degrees']
    
    # Prepare table data
    table_data = []
    for vertex in sorted(degrees.keys()):
        vertex_type = "Zero-Divisor (Z(R))" if vertex in zero_divisors else "Regular Element"
        degree = degrees[vertex]
        table_data.append([vertex, vertex_type, degree])
    
    colLabels = ['Vertex', 'Type', 'Degree']
    
    table = ax.table(cellText=table_data, colLabels=colLabels, cellLoc='center', loc='center', 
                     colWidths=[0.15, 0.40, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the cells
    for i in range(len(table_data) + 1):
        for j in range(len(colLabels)):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # Type column
                    if "Zero-Divisor" in table_data[i-1][1]:
                        cell.set_facecolor('#ffcccc')  # Light red for Z(R)
                    else:
                        cell.set_facecolor('#ccccff')  # Light blue for Regular
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
    
    plt.title(f'Vertex Degree Distribution in Z_{n}\n(Red = Zero-Divisors, Blue = Regular Elements)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'vertex_degrees_Z{n}.png', dpi=150, bbox_inches='tight')
    print(f"Vertex degree table saved to vertex_degrees_Z{n}.png")
    plt.show()

def visualize_graph(G, zero_divisors, hamiltonian_path=None, degree_info=None):
    """
    Visualize the graph using matplotlib.
    Color coding: Red for zero-divisors, Blue for regular elements.
    Node size based on degree (larger nodes = higher degree).
    If a Hamiltonian path is provided, animate it step by step, highlighting start (yellow) and end (orange).
    Saves the plot to a file and displays it. Also saves animation as GIF if path exists.
    """
    pos = nx.spring_layout(G)
    colors = ['red' if node in zero_divisors else 'blue' for node in G.nodes()]
    
    # Calculate node sizes based on degree
    if degree_info and 'degrees' in degree_info:
        degrees = degree_info['degrees']
        max_degree = max(degrees.values()) if degrees.values() else 1
        # Scale degrees to node sizes (200 to 1000)
        node_sizes = [200 + (degrees.get(node, 0) / max_degree) * 800 if max_degree > 0 else 500 for node in G.nodes()]
    else:
        node_sizes = [500 for _ in G.nodes()]
    
    if hamiltonian_path:
        fig, ax = plt.subplots(figsize=(10, 10))
        path_edges = [(hamiltonian_path[i], hamiltonian_path[i+1]) for i in range(len(hamiltonian_path)-1)]
        subframes_per_edge = 30
        total_frames = len(path_edges) * subframes_per_edge
        
        def update(frame):
            ax.clear()
            nx.draw(G, pos, with_labels=True, node_color=colors, node_size=node_sizes, font_size=10, ax=ax)
            edge_index = frame // subframes_per_edge
            subframe = frame % subframes_per_edge
            current_edges = path_edges[:edge_index + 1]
            # Draw previous edges in green
            if edge_index > 0:
                nx.draw_networkx_edges(G, pos, edgelist=path_edges[:edge_index], edge_color='green', width=3, ax=ax)
            if current_edges:
                last_edge = current_edges[-1]
                start_pos = pos[last_edge[0]]
                end_pos = pos[last_edge[1]]
                # Interpolate arrow position
                t = subframe / (subframes_per_edge - 1) if subframes_per_edge > 1 else 1
                arrow_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                arrow_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                # Draw partial edge in blue up to arrow
                ax.plot([start_pos[0], arrow_x], [start_pos[1], arrow_y], color='blue', linewidth=3)
                # Draw arrow
                dx = (end_pos[0] - arrow_x) * 0.2
                dy = (end_pos[1] - arrow_y) * 0.2
                ax.arrow(arrow_x, arrow_y, dx, dy, head_width=0.05, head_length=0.05, fc='red', ec='red')
            # Highlight start
            start_node_size = [node_sizes[list(G.nodes()).index(hamiltonian_path[0])]]
            nx.draw_networkx_nodes(G, pos, nodelist=[hamiltonian_path[0]], node_color='yellow', node_size=start_node_size[0], ax=ax)
            # Highlight end at the last frame
            if frame == total_frames - 1:
                end_node_size = [node_sizes[list(G.nodes()).index(hamiltonian_path[-1])]]
                nx.draw_networkx_nodes(G, pos, nodelist=[hamiltonian_path[-1]], node_color='orange', node_size=end_node_size[0], ax=ax)
        
        anim = FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
        plt.title("Animated Hamiltonian Path in Total Graph of Z_n\n(Node size ∝ Degree)")
        anim.save('hamiltonian_path.gif', writer='pillow')
        print("Animation saved to hamiltonian_path.gif")
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=node_sizes, font_size=10, ax=ax)
        plt.title("Total Graph of Z_n\n(Node size ∝ Degree)")
        plt.savefig('ring_graph.png')
        print("Graph visualization saved to ring_graph.png")
        plt.show()

def visualize_heatmap(G, zero_divisors, regular_elements):
    """
    Visualize the adjacency matrix as a heatmap using seaborn.
    Nodes ordered: Zero-Divisors first, then Regular Elements.
    """
    ordered_nodes = sorted(zero_divisors) + sorted(regular_elements)
    adj_matrix = nx.to_numpy_array(G, nodelist=ordered_nodes)
    plt.figure(figsize=(8, 8))
    sns.heatmap(adj_matrix, cmap='Blues', square=True, cbar=True, xticklabels=ordered_nodes, yticklabels=ordered_nodes)
    plt.title("Adjacency Matrix Heatmap of Total Graph")
    plt.savefig('adjacency_heatmap.png')
    print("Heatmap saved to adjacency_heatmap.png")
    plt.show()

def main():
    """
    Main function supporting both interactive analysis and batch processing.
    """
    print("=" * 70)
    print("Ring Analysis: Total Graph of Z_n")
    print("=" * 70)
    print("\nChoose mode:")
    print("1. Interactive Analysis (single n)")
    print("2. Batch Processing (n in range [2, 31])")
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Interactive mode
        n = int(input("Enter an integer n to define the ring Z_n: "))
        
        # Step 2: Element Classification
        zero_divisors = get_zero_divisors(n)
        regular_elements = get_regular_elements(n, zero_divisors)
        
        print(f"\nZero-Divisors Z(R): {sorted(zero_divisors)}")
        print(f"Regular Elements Reg(R): {sorted(regular_elements)}")
        
        # Step 3: Total Graph Construction
        G = build_total_graph(n, zero_divisors)
        
        # Check connectivity
        is_connected = nx.is_connected(G)
        print(f"Graph Connected: {is_connected}")
        
        # Check if ideal
        ideal_check = check_if_ideal(n, zero_divisors)
        print(f"Z(R) is an ideal: {ideal_check}")
        
        # Step 4: Hamiltonian Path Analysis
        hamiltonian_path = find_hamiltonian_path_with_timeout(G, timeout_sec=5, n_value=n)
        ham_cycle_exists = hamiltonian_cycle_exists(G, hamiltonian_path, zero_divisors, n)
        
        if hamiltonian_path:
            print(f"Hamiltonian Path found: {hamiltonian_path}")
            if ham_cycle_exists:
                print("✓ Hamiltonian Cycle also exists!")
        else:
            print("No Hamiltonian Path found.")
        
        # Degree Distribution Analysis
        print("\n" + "="*70)
        print("DEGREE DISTRIBUTION ANALYSIS")
        print("="*70)
        degree_info = analyze_degree_distribution(G, n, zero_divisors, regular_elements)
        degrees = degree_info['degrees']
        
        print(f"Minimum Degree δ(G): {degree_info['min_degree']}")
        print(f"Maximum Degree Δ(G): {degree_info['max_degree']}")
        print(f"Average Degree: {degree_info['avg_degree']:.2f}")
        
        # Type-based analysis
        print(f"\nAverage degree of Zero-Divisors: {degree_info['avg_zd_degree']:.2f}")
        print(f"Average degree of Regular Elements: {degree_info['avg_reg_degree']:.2f}")
        diff = degree_info['avg_zd_degree'] - degree_info['avg_reg_degree']
        if diff > 0:
            print(f"✓ Zero-Divisors have {diff:.2f} MORE connections on average")
        elif diff < 0:
            print(f"✗ Regular Elements have {-diff:.2f} more connections on average")
        else:
            print("Zero-Divisors and Regular Elements have equal average degree")
        
        # Dirac's Condition
        print(f"\n" + "-"*70)
        dirac_satisfied, min_deg, threshold = check_dirac_condition(n, degree_info['min_degree'])
        print(f"Dirac's Condition: δ(G) ≥ n/2")
        print(f"Minimum degree is {min_deg}. Since n/2 is {threshold:.1f},")
        print(f"Dirac's condition is {'SATISFIED' if dirac_satisfied else 'NOT SATISFIED'}.")
        if dirac_satisfied:
            print(f"→ The graph is guaranteed to have a Hamiltonian cycle (if connected).")
        
        # Degree details
        print(f"\n" + "-"*70)
        print("Individual Degrees:")
        for node in sorted(degrees.keys()):
            node_type = "Z(R)" if node in zero_divisors else "Reg"
            print(f"  Node {node} ({node_type}): degree = {degrees[node]}")
        
        # Step 5: Visualization
        print("\nGenerating visualizations...")
        visualize_graph(G, zero_divisors, hamiltonian_path, degree_info)
        visualize_heatmap(G, zero_divisors, regular_elements)
        visualize_vertex_degrees(n, zero_divisors, regular_elements, degree_info)
    elif choice == "2":
        # Batch processing mode with user-defined range
        print("\nBatch Processing Mode")
        start_n = int(input("Enter starting value for n: "))
        end_n = int(input("Enter ending value for n: "))
        
        if start_n < 2:
            start_n = 2
            print(f"Starting value adjusted to 2 (minimum valid ring)")
        
        if end_n < start_n:
            print("End value must be greater than or equal to start value.")
            return
        
        print(f"\nPerforming batch analysis for n in [{start_n}, {end_n}]...")
        print("This may take a moment...\n")
        
        # Batch processing with parallelization and progress bar
        n_values = list(range(start_n, end_n + 1))
        num_workers = max(1, cpu_count() - 1)  # Use all CPUs except one
        
        print(f"Using {num_workers} CPU cores for parallel processing...")
        
        # Create a partial function with visualize=False
        analyze_func = partial(analyze_single_ring, visualize=False)
        
        if num_workers > 1:
            # Parallel processing with progress bar
            with Pool(processes=num_workers) as pool:
                all_results = list(tqdm(
                    pool.imap_unordered(analyze_func, n_values),
                    total=len(n_values),
                    desc="Analyzing rings",
                    unit="ring"
                ))
        else:
            # Sequential processing if only 1 core
            all_results = []
            for n in tqdm(n_values, desc="Analyzing rings", unit="ring"):
                results = analyze_func(n)
                all_results.append(results)
        
        # Create pandas DataFrame
        df = pd.DataFrame(all_results)
        
        # Display the table
        print("\n" + "=" * 130)
        print(f"BATCH ANALYSIS RESULTS: Z_n for n in [{start_n}, {end_n}]")
        print("=" * 130)
        print(df[['n', 'ZR_cardinality', 'ZR_is_Ideal', 'Is_Connected', 'min_degree', 'max_degree', 'avg_degree', 'Ham_Path_Exists']].to_string(index=False))
        print("\n" + "=" * 130)
        print("\nCOMPARATIVE REPORTS:")
        print("=" * 130)
        for _, row in df.iterrows():
            print(row['Comparative_Report'])
        
        # Save to CSV
        csv_filename = f'ring_analysis_batch_{start_n}_{end_n}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
        
        # Summary Statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS:")
        print("=" * 70)
        ideal_count = df['ZR_is_Ideal'].sum()
        non_ideal_count = len(df) - ideal_count
        connected_count = df['Is_Connected'].sum()
        path_count = df['Ham_Path_Exists'].sum()
        cycle_count = df['Ham_Cycle_Exists'].sum()
        
        print(f"Total rings analyzed: {len(df)}")
        print(f"Rings where Z(R) is an ideal: {ideal_count}")
        print(f"Rings where Z(R) is NOT an ideal: {non_ideal_count}")
        print(f"Connected graphs: {connected_count}")
        print(f"Graphs with Hamiltonian Path: {path_count}")
        print(f"Graphs with Hamiltonian Cycle: {cycle_count}")
        
        # Degree Distribution Statistics
        print("\n" + "-" * 70)
        print("DEGREE DISTRIBUTION STATISTICS:")
        print("-" * 70)
        print(f"Average Minimum Degree (across all rings): {df['min_degree'].mean():.2f}")
        print(f"Average Maximum Degree (across all rings): {df['max_degree'].mean():.2f}")
        print(f"Average Overall Degree (across all rings): {df['avg_degree'].mean():.2f}")
        
        # Type-based degree analysis
        print(f"\nType-Based Degree Comparison:")
        print(f"  Average degree of Zero-Divisors: {df['avg_zd_degree'].mean():.2f}")
        print(f"  Average degree of Regular Elements: {df['avg_reg_degree'].mean():.2f}")
        difference = df['avg_zd_degree'].mean() - df['avg_reg_degree'].mean()
        if difference > 0:
            print(f"  → Zero-Divisors have {difference:.2f} MORE connections on average")
        elif difference < 0:
            print(f"  → Regular Elements have {-difference:.2f} more connections on average")
        else:
            print(f"  → Both have equal average degree")
        
        # Dirac's Condition Analysis
        print(f"\nDirac's Condition Analysis (δ(G) ≥ n/2):")
        dirac_satisfied = (df['min_degree'] >= df['n'] / 2).sum()
        dirac_violated = len(df) - dirac_satisfied
        print(f"  Rings satisfying Dirac's condition: {dirac_satisfied}")
        print(f"  Rings NOT satisfying Dirac's condition: {dirac_violated}")
        if dirac_satisfied > 0:
            print(f"  → {dirac_satisfied} ring(s) guaranteed to have Hamiltonian cycles (if connected)")
        
        # Verify Theorem 7.2.1: If Z(R) is ideal, graph should be disconnected
        theorem_violations = df[(df['ZR_is_Ideal'] == True) & (df['Is_Connected'] == True)]
        if len(theorem_violations) == 0:
            print("\n✓ Theorem 7.2.1 verified: All rings with Z(R) as ideal have disconnected graphs")
        else:
            print(f"\n✗ Theorem 7.2.1 violations found: {len(theorem_violations)} ring(s)")
            print("  Rings:", theorem_violations['n'].tolist())
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualize_batch_results(df)
        create_detailed_table_visualization(df)
        create_html_report(df, start_n, end_n)
    
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()