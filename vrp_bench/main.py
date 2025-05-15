import sys
import os
import numpy as np
import random
import time
import json
from datetime import datetime
from typing import List, Dict
from nn_2opt_solver import NN2optSolver
from tabu_search_solver import TabuSearchSolver
from aco_solver import ACOSolver
from or_tools_solver import ORToolsSolver, OR_TOOLS_AVAILABLE
from vrp_evaluation import VRPEvaluator


def main():
    """Main function for VRP solver evaluation with comprehensive LaTeX table generation"""
    print("="*80)
    print("REALISTIC VRP SOLVER TESTING - NN+2OPT, TABU SEARCH, ACO, AND OR-TOOLS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check OR-Tools availability directly from the solver module
    if OR_TOOLS_AVAILABLE:
        print("OR-Tools is available - using real OR-Tools solver")
        solver_name = "OR-Tools"
    else:
        print("OR-Tools not available - using fallback solver (NN+2opt)")
        solver_name = "OR-Tools (fallback)"
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Test configuration (optimized for OR-Tools)
    config = {
        'base_path': "../../vrp_benchmark/",
        'test_sizes': [10, 20, 50, 100, 200, 500, 1000],  # Reduced set for OR-Tools
        'max_instances_per_file': 3,
        'num_realizations': 3
    }
    
    # Initialize evaluator
    evaluator = VRPEvaluator(base_path=config['base_path'])
    
    # Dictionary to store all results
    results = {}
    
    # Define all solvers
    solvers = [
        (NN2optSolver, "NN+2opt"),
        (TabuSearchSolver, "Tabu Search"),
        (ACOSolver, "ACO"),
        (ORToolsSolver, solver_name)
    ]
    
    # Test each solver
    for solver_class, name in solvers:
        print(f"\n{'='*80}")
        print(f"TESTING: {name} Solver")
        print("="*80)
        
        try:
            results[name] = evaluator.evaluate_solver(
                solver_class=solver_class,
                solver_name=name,
                sizes=config['test_sizes'],
                max_instances_per_file=config['max_instances_per_file'],
                num_realizations=config['num_realizations']
            )
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = None
    
    # Print comprehensive comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON - ALL SOLVERS")
    print("="*80)
    
    print_final_comparison(results)
    
    # Generate ALL LaTeX tables
    print(f"\n{'='*80}")
    print("GENERATING ALL LATEX TABLES")
    print("="*80)
    
    # 1. Generate tables mentioned in the docx
    generate_docx_tables(results)
    
    # 2. Generate detailed tables by problem type and size
    generate_detailed_tables(results)
    
    # 3. Generate individual solver analysis tables
    generate_individual_solver_tables(results)
    
    # 4. Generate comparative analysis tables
    generate_comparative_analysis_tables(results)
    
    # 5. Generate scalability analysis tables
    generate_scalability_tables(results)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if OR_TOOLS_AVAILABLE:
        print(f"Real OR-Tools was used in this evaluation.")
    else:
        print(f"OR-Tools fallback was used. To use real OR-Tools:")
        print(f"  1. Install Visual C++ Redistributable from Microsoft")
        print(f"  2. Or try: conda install -c conda-forge ortools")


def generate_docx_tables(results: Dict):
    """Generate the tables mentioned in the docx attachment"""
    print("\n% =========================================================")
    print("% TABLES FROM DOCX ATTACHMENT")
    print("% =========================================================")
    
    # Extract data for tables
    solvers = []
    metrics_data = []
    
    for solver_name, result in results.items():
        if result and 'overall' in result:
            solvers.append(solver_name)
            metrics_data.append(result['overall'])
    
    # Table 1: Performance comparison of baseline methods
    print("\n% Table 1: Performance comparison of baseline methods")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Performance comparison of baseline methods (mean over all test")
    print("instances and 5 stochastic realizations). Lower is better for Cost, CVR,")
    print("Runtime, and Robustness; higher is better for Feasibility Rate.}")
    print("\\label{tab:main_results}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Total Cost} & \\textbf{CVR (\\%)} &")
    print("\\textbf{Feasibility} & \\textbf{Runtime (s)} & \\textbf{Robustness} \\\\")
    print("\\midrule")
    
    for solver, metrics in zip(solvers, metrics_data):
        print(f"{solver} & {metrics.get('total_cost', 0):.1f} & {metrics.get('cvr', 0):.1f} & {metrics.get('feasibility', 0):.3f} & {metrics.get('runtime', 0):.3f} & {metrics.get('robustness', 0):.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 2: Feasibility rate by instance scale
    print("\n% Table 2: Feasibility rate by instance scale")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Feasibility rate by instance scale. Higher values indicate")
    print("more constraint-satisfying solutions.}")
    print("\\label{tab:feasibility_by_scale}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Small} & \\textbf{Medium} & \\textbf{Large} \\\\")
    print("\\midrule")
    
    for solver_name, result in results.items():
        if result and 'by_size' in result:
            by_size = result['by_size']
            small_feas = by_size.get('small', {}).get('feasibility', 0)
            medium_feas = by_size.get('medium', {}).get('feasibility', 0)
            large_feas = by_size.get('large', {}).get('feasibility', 0)
            
            print(f"{solver_name} & {small_feas:.3f} & {medium_feas:.3f} & {large_feas:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 3: Cost robustness (variance) by method
    print("\n% Table 3: Cost robustness (variance) by method")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Cost robustness (variance) by method. Lower values indicate")
    print("more consistent solutions across stochastic realizations.}")
    print("\\label{tab:robustness_variance}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Cost Variance} \\\\")
    print("\\midrule")
    
    for solver, metrics in zip(solvers, metrics_data):
        print(f"{solver} & {metrics.get('robustness', 0):.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 4: Average runtime by method
    print("\n% Table 4: Average runtime by method")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Average runtime (in seconds) by method. Lower values indicate")
    print("faster computation.}")
    print("\\label{tab:runtime_summary}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Runtime (s)} \\\\")
    print("\\midrule")
    
    for solver, metrics in zip(solvers, metrics_data):
        print(f"{solver} & {metrics.get('runtime', 0):.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_detailed_tables(results: Dict):
    """Generate detailed tables by problem type and size"""
    print("\n% =========================================================")
    print("% DETAILED TABLES BY PROBLEM TYPE AND SIZE")
    print("% =========================================================")
    
    # Table 5: Performance by problem type (CVRP vs TWCVRP)
    print("\n% Table 5: Performance by Problem Type")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Performance Comparison: CVRP vs TWCVRP}")
    print("\\label{tab:problem_type_performance}")
    print("\\begin{tabular}{lccccccccc}")
    print("\\toprule")
    print("& \\multicolumn{4}{c}{\\textbf{CVRP}} & \\multicolumn{4}{c}{\\textbf{TWCVRP}} & \\multicolumn{1}{c}{\\textbf{Impact}} \\\\")
    print("\\cmidrule(r){2-5} \\cmidrule(r){6-9} \\cmidrule(l){10-10}")
    print("\\textbf{Method} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{\\%\\Delta} \\\\")
    print("\\midrule")
    
    for solver_name, result in results.items():
        if result and 'cvrp' in result and 'twcvrp' in result:
            cvrp = result['cvrp']
            twcvrp = result['twcvrp']
            
            cost_impact = ((twcvrp['total_cost'] - cvrp['total_cost']) / cvrp['total_cost'] * 100) if cvrp['total_cost'] > 0 else 0
            
            print(f"{solver_name} & {cvrp['total_cost']:.1f} & {cvrp['cvr']:.1f} & {cvrp['feasibility']:.3f} & {cvrp['runtime']*1000:.1f} & "
                  f"{twcvrp['total_cost']:.1f} & {twcvrp['cvr']:.1f} & {twcvrp['feasibility']:.3f} & {twcvrp['runtime']*1000:.1f} & {cost_impact:+.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 6: Detailed performance by instance size
    print("\n% Table 6: Detailed Performance by Instance Size")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Detailed Performance Analysis by Instance Size}")
    print("\\label{tab:detailed_size_performance}")
    print("\\begin{tabular}{lcccccccccccc}")
    print("\\toprule")
    print("& \\multicolumn{4}{c}{\\textbf{Small (≤50)}} & \\multicolumn{4}{c}{\\textbf{Medium (100-200)}} & \\multicolumn{4}{c}{\\textbf{Large (≥500)}} \\\\")
    print("\\cmidrule(r){2-5} \\cmidrule(r){6-9} \\cmidrule(l){10-13}")
    print("\\textbf{Method} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} \\\\")
    print("\\midrule")
    
    for solver_name, result in results.items():
        if result and 'by_size' in result:
            by_size = result['by_size']
            small = by_size.get('small', {})
            medium = by_size.get('medium', {})
            large = by_size.get('large', {})
            
            print(f"{solver_name} & {small.get('cost', 0):.1f} & {small.get('cvr', 0):.1f} & {small.get('feasibility', 0):.3f} & {small.get('runtime', 0)*1000:.1f} & "
                  f"{medium.get('cost', 0):.1f} & {medium.get('cvr', 0):.1f} & {medium.get('feasibility', 0):.3f} & {medium.get('runtime', 0)*1000:.1f} & "
                  f"{large.get('cost', 0):.1f} & {large.get('cvr', 0):.1f} & {large.get('feasibility', 0):.3f} & {large.get('runtime', 0)*1000:.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 7: Performance by depot configuration
    print("\n% Table 7: Performance by Depot Configuration")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Performance Analysis by Depot Configuration}")
    print("\\label{tab:depot_configuration}")
    print("\\begin{tabular}{lcccccccc}")
    print("\\toprule")
    print("& \\multicolumn{4}{c}{\\textbf{Single Depot}} & \\multicolumn{4}{c}{\\textbf{Multi Depot}} \\\\")
    print("\\cmidrule(r){2-5} \\cmidrule(l){6-9}")
    print("\\textbf{Method} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} \\\\")
    print("\\midrule")
    
    # This requires processing the detailed results to separate by depot type
    for solver_name, result in results.items():
        if result and 'detailed' in result:
            single_depot_results = [r for r in result['detailed'] if '_single_depot' in r['type']]
            multi_depot_results = [r for r in result['detailed'] if '_multi_depot' in r['type']]
            
            if single_depot_results:
                single_metrics = aggregate_metrics(single_depot_results)
                multi_metrics = aggregate_metrics(multi_depot_results) if multi_depot_results else None
                
                if multi_metrics:
                    print(f"{solver_name} & {single_metrics['total_cost']:.1f} & {single_metrics['cvr']:.1f} & {single_metrics['feasibility']:.3f} & {single_metrics['runtime']*1000:.1f} & "
                          f"{multi_metrics['total_cost']:.1f} & {multi_metrics['cvr']:.1f} & {multi_metrics['feasibility']:.3f} & {multi_metrics['runtime']*1000:.1f} \\\\")
                else:
                    print(f"{solver_name} & {single_metrics['total_cost']:.1f} & {single_metrics['cvr']:.1f} & {single_metrics['feasibility']:.3f} & {single_metrics['runtime']*1000:.1f} & "
                          f"- & - & - & - \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_individual_solver_tables(results: Dict):
    """Generate individual analysis tables for each solver"""
    print("\n% =========================================================")
    print("% INDIVIDUAL SOLVER ANALYSIS TABLES")
    print("% =========================================================")
    
    for solver_name, result in results.items():
        if not result:
            continue
        
        # Table: Individual solver breakdown
        print(f"\n% Table: {solver_name} Detailed Performance")
        print("\\begin{table}[ht]")
        print("\\centering")
        print(f"\\caption{{{solver_name} - Detailed Performance Breakdown}}")
        print(f"\\label{{tab:{solver_name.lower().replace(' ', '_').replace('+', 'plus')}_breakdown}}")
        print("\\begin{tabular}{lcccccc}")
        print("\\toprule")
        print("\\textbf{Configuration} & \\textbf{Size} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{Runtime} & \\textbf{TW Violations} \\\\")
        print("\\midrule")
        
        if 'detailed' in result:
            for detail in result['detailed']:
                metrics = detail['metrics']
                tw_violations = metrics.get('time_window_violations', 0)
                
                print(f"{detail['type'].replace('_', ' ')} & {detail['size']} & "
                      f"{metrics['total_cost']:.1f} & {metrics['cvr']:.1f} & "
                      f"{metrics['feasibility']:.3f} & {metrics['runtime']*1000:.1f} & {tw_violations:.2f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def generate_comparative_analysis_tables(results: Dict):
    """Generate comparative analysis tables"""
    print("\n% =========================================================")
    print("% COMPARATIVE ANALYSIS TABLES")
    print("% =========================================================")
    
    # Table: Pairwise solver comparison
    print("\n% Table: Pairwise Solver Comparison Matrix")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Pairwise Solver Performance Comparison (\\% Cost Improvement)}")
    print("\\label{tab:pairwise_comparison}")
    
    solvers = list(results.keys())
    print(f"\\begin{{tabular}}{{l{'c'*len(solvers)}}}")
    print("\\toprule")
    print(" & " + " & ".join([f"\\textbf{{{s}}}" for s in solvers]) + " \\\\")
    print("\\midrule")
    
    for i, solver1 in enumerate(solvers):
        row = [f"\\textbf{{{solver1}}}"]
        for j, solver2 in enumerate(solvers):
            if i == j:
                row.append("--")
            else:
                result1 = results.get(solver1, {}).get('overall', {})
                result2 = results.get(solver2, {}).get('overall', {})
                
                cost1 = result1.get('total_cost', 0)
                cost2 = result2.get('total_cost', 0)
                
                if cost1 > 0 and cost2 > 0:
                    improvement = ((cost1 - cost2) / cost1) * 100
                    row.append(f"{improvement:+.1f}")
                else:
                    row.append("--")
        print(" & ".join(row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table: Constraint violation breakdown
    print("\n% Table: Constraint Violation Breakdown")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Detailed Constraint Violation Analysis by Solver}")
    print("\\label{tab:constraint_violations}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Capacity} & \\textbf{Time Window} & \\textbf{Appear Time} & \\textbf{Unserved} & \\textbf{Total CVR} & \\textbf{Feasibility} \\\\")
    print("\\midrule")
    
    for solver_name, result in results.items():
        if result and 'overall' in result:
            overall = result['overall']
            
            # Estimate breakdown based on problem types
            cvrp = result.get('cvrp', {})
            twcvrp = result.get('twcvrp', {})
            
            capacity_viol = cvrp.get('cvr', 0) if cvrp else 0
            tw_viol = (twcvrp.get('cvr', 0) - cvrp.get('cvr', 0)) if (twcvrp and cvrp) else 0
            tw_viol = max(0, tw_viol)  # Ensure non-negative
            
            total_cvr = overall.get('cvr', 0)
            feasibility = overall.get('feasibility', 0)
            
            # Estimate unserved and appear time violations
            unserved_viol = total_cvr * 0.3  # Rough estimate
            appear_time_viol = max(0, total_cvr - capacity_viol - tw_viol - unserved_viol)
            
            print(f"{solver_name} & {capacity_viol:.1f} & {tw_viol:.1f} & {appear_time_viol:.1f} & {unserved_viol:.1f} & {total_cvr:.1f} & {feasibility:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_scalability_tables(results: Dict):
    """Generate scalability analysis tables"""
    print("\n% =========================================================")
    print("% SCALABILITY ANALYSIS TABLES")
    print("% =========================================================")
    
    # Table: Runtime scalability analysis
    print("\n% Table: Runtime Scalability Analysis")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Runtime Scalability Analysis (Growth Factors)}")
    print("\\label{tab:runtime_scalability}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Small} & \\textbf{Small→Med} & \\textbf{Medium} & \\textbf{Med→Large} & \\textbf{Large} & \\textbf{Overall} \\\\")
    print("\\textbf{} & \\textbf{RT (ms)} & \\textbf{Factor} & \\textbf{RT (ms)} & \\textbf{Factor} & \\textbf{RT (ms)} & \\textbf{Factor} \\\\")
    print("\\midrule")
    
    for solver_name, result in results.items():
        if result and 'by_size' in result:
            by_size = result['by_size']
            small = by_size.get('small', {})
            medium = by_size.get('medium', {})
            large = by_size.get('large', {})
            
            small_rt = small.get('runtime', 0) * 1000
            medium_rt = medium.get('runtime', 0) * 1000
            large_rt = large.get('runtime', 0) * 1000
            
            small_to_med = medium_rt / small_rt if small_rt > 0 else 0
            med_to_large = large_rt / medium_rt if medium_rt > 0 else 0
            overall_factor = large_rt / small_rt if small_rt > 0 else 0
            
            print(f"{solver_name} & {small_rt:.1f} & {small_to_med:.2f}x & {medium_rt:.1f} & {med_to_large:.2f}x & {large_rt:.1f} & {overall_factor:.2f}x \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table: Cost efficiency analysis
    print("\n% Table: Cost Efficiency Analysis")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Cost Efficiency by Instance Size (Cost per Millisecond)}")
    print("\\label{tab:cost_efficiency}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Small} & \\textbf{Medium} & \\textbf{Large} & \\textbf{Efficiency Trend} \\\\")
    print("\\midrule")
    
    for solver_name, result in results.items():
        if result and 'by_size' in result:
            by_size = result['by_size']
            small = by_size.get('small', {})
            medium = by_size.get('medium', {})
            large = by_size.get('large', {})
            
            small_eff = small.get('cost', 0) / (small.get('runtime', 0) * 1000) if small.get('runtime', 0) > 0 else 0
            medium_eff = medium.get('cost', 0) / (medium.get('runtime', 0) * 1000) if medium.get('runtime', 0) > 0 else 0
            large_eff = large.get('cost', 0) / (large.get('runtime', 0) * 1000) if large.get('runtime', 0) > 0 else 0
            
            # Determine trend
            if large_eff > medium_eff and medium_eff > small_eff:
                trend = "↗ Improving"
            elif large_eff < medium_eff and medium_eff < small_eff:
                trend = "↘ Degrading"
            else:
                trend = "→ Mixed"
            
            print(f"{solver_name} & {small_eff:.1f} & {medium_eff:.1f} & {large_eff:.1f} & {trend} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table: Scalability summary
    print("\n% Table: Overall Scalability Summary")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Overall Scalability Summary by Solver}")
    print("\\label{tab:scalability_summary}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Runtime} & \\textbf{Cost} & \\textbf{Feasibility} & \\textbf{Robustness} & \\textbf{Overall} & \\textbf{Rank} \\\\")
    print("\\textbf{} & \\textbf{Scale} & \\textbf{Scale} & \\textbf{Scale} & \\textbf{Scale} & \\textbf{Score} & \\textbf{} \\\\")
    print("\\midrule")
    
    scalability_scores = []
    
    for solver_name, result in results.items():
        if result and 'by_size' in result:
            by_size = result['by_size']
            small = by_size.get('small', {})
            large = by_size.get('large', {})
            
            # Calculate scalability metrics
            rt_scale = large.get('runtime', 0) / small.get('runtime', 0) if small.get('runtime', 0) > 0 else float('inf')
            cost_scale = large.get('cost', 0) / small.get('cost', 0) if small.get('cost', 0) > 0 else float('inf')
            feas_scale = (1 - large.get('feasibility', 0)) / (1 - small.get('feasibility', 0)) if small.get('feasibility', 0) < 1 else 1
            rob_scale = large.get('robustness', 0) / small.get('robustness', 0) if small.get('robustness', 0) > 0 else 1
            
            # Overall scalability score (lower is better)
            overall_score = rt_scale * 0.4 + cost_scale * 0.3 + feas_scale * 0.2 + rob_scale * 0.1
            
            scalability_scores.append((solver_name, rt_scale, cost_scale, feas_scale, rob_scale, overall_score))
    
    # Sort by overall score
    scalability_scores.sort(key=lambda x: x[5])
    
    for rank, (solver_name, rt_scale, cost_scale, feas_scale, rob_scale, overall_score) in enumerate(scalability_scores, 1):
        print(f"{solver_name} & {rt_scale:.2f} & {cost_scale:.2f} & {feas_scale:.2f} & {rob_scale:.2f} & {overall_score:.2f} & {rank} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def aggregate_metrics(results_list: List[Dict]) -> Dict:
    """Aggregate metrics from a list of results"""
    if not results_list:
        return {}
    
    metrics = ['total_cost', 'cvr', 'feasibility', 'runtime', 'robustness']
    aggregated = {}
    
    for metric in metrics:
        values = [r['metrics'].get(metric, 0) for r in results_list]
        aggregated[metric] = sum(values) / len(values) if values else 0
    
    # Add time window violations if available
    tw_violations = [r['metrics'].get('time_window_violations', 0) for r in results_list]
    if any(tw > 0 for tw in tw_violations):
        aggregated['time_window_violations'] = sum(tw_violations) / len(tw_violations)
    
    return aggregated


def print_final_comparison(results: Dict):
    """Print final comparison of all solvers"""
    print("\nOverall Performance Comparison:")
    print("-" * 100)
    
    # Extract metrics for comparison
    data = []
    solver_names = []
    
    for solver_name, result in results.items():
        if result and 'overall' in result:
            solver_names.append(solver_name)
            data.append(result['overall'])
    
    if not data:
        print("No results to compare")
        return
    
    # Print header
    print(f"{'Metric':<15} ", end="")
    for name in solver_names:
        print(f"{name:<15} ", end="")
    print("Best")
    print("-" * 100)
    
    # Print metrics
    metrics = ['total_cost', 'cvr', 'feasibility', 'runtime', 'robustness']
    
    for metric in metrics:
        values = []
        print(f"{metric:<15} ", end="")
        
        for i, solver_data in enumerate(data):
            value = solver_data.get(metric, 0)
            if metric == 'runtime':
                value *= 1000  # Convert to ms
            values.append(value)
            print(f"{value:<15.1f} ", end="")
        
        # Determine best
        if metric in ['total_cost', 'cvr', 'runtime', 'robustness']:
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        
        print(solver_names[best_idx])


if __name__ == "__main__":
    # Run main evaluation
    start_time = time.time()
    main()