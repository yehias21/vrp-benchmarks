import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List


class VRPEvaluator:
    """Evaluation framework for VRP solvers (optimized for large instances)"""
    
    def __init__(self, base_path: str = "../../vrp_benchmark/"):
        self.base_path = base_path
        self.problems = ['real_cvrp/cvrp_', 'real_twcvrp/twvrp_']
        self.types = [['_single_depot_single_vehicule_sumDemands', '_multi_depot'], 
                     ['_depots_equal_city', '_single_depot']]
        self.size_categories = {
            'small': [10, 20, 50],
            'medium': [100, 200],
            'large': [500, 1000]
        }
    
    def evaluate_solver(self, solver_class, solver_name: str, sizes: List[int] = [10, 20, 50, 100],
                       max_instances_per_file: int = 5, num_realizations: int = 10) -> Dict:
        """Evaluate a solver on the benchmark suite with adaptive parameters"""
        all_results = []
        results_by_size = defaultdict(list)
        
        print(f"Starting evaluation of {solver_name}...")
        
        instance_count = 0
        
        for i in range(len(self.problems)):
            for size in sizes:
                # Adjust parameters based on problem size
                if size >= 500:
                    actual_instances = min(max_instances_per_file, 1)
                    actual_realizations = 1
                elif size >= 200:
                    actual_instances = min(max_instances_per_file, 2)
                    actual_realizations = min(num_realizations, 2)
                elif size >= 100:
                    actual_instances = min(max_instances_per_file, 3)
                    actual_realizations = min(num_realizations, 3)
                else:
                    actual_instances = max_instances_per_file
                    actual_realizations = num_realizations
                
                print(f"Processing size {size}: {actual_instances} instances, {actual_realizations} realizations")
                
                for type_variant in self.types[i]:
                    data_path = self.base_path + self.problems[i] + str(size) + type_variant + '.npz'
                    
                    if not os.path.exists(data_path):
                        continue
                    
                    try:
                        # Load data
                        data = np.load(data_path, allow_pickle=True)
                        data_dict = self._convert_to_dict(data)
                        
                        # Limit instances for efficiency
                        limited_data = self._limit_instances(data_dict, actual_instances)
                        
                        # Skip empty datasets
                        if self._is_empty_dataset(limited_data):
                            continue
                        
                        # Create solver instance
                        solver = solver_class(limited_data)
                        
                        # Solve all instances
                        avg_results, instance_results = solver.solve_all_instances(actual_realizations)
                        
                        # Store results
                        result_entry = {
                            'problem': self.problems[i],
                            'size': size,
                            'type': type_variant,
                            'metrics': avg_results,
                            'problem_type': 'twcvrp' if 'twcvrp' in self.problems[i] else 'cvrp'
                        }
                        all_results.append(result_entry)
                        
                        # Categorize by size
                        for category, size_list in self.size_categories.items():
                            if size in size_list:
                                results_by_size[category].append(avg_results)
                        
                        instance_count += 1
                        print(f"Completed {solver_name} - size {size}: Cost={avg_results['total_cost']:.1f}, "
                              f"Runtime={avg_results['runtime']*1000:.1f}ms")
                    
                    except Exception as e:
                        print(f"Error processing {data_path}: {e}")
                        continue
        
        print(f"Completed evaluation: {instance_count} problem-type combinations processed")
        
        # Aggregate results
        overall_metrics = self._aggregate_results(all_results)
        size_metrics = self._aggregate_size_results(results_by_size)
        
        # Separate CVRP and TWCVRP results
        cvrp_results = [r for r in all_results if r['problem_type'] == 'cvrp']
        twcvrp_results = [r for r in all_results if r['problem_type'] == 'twcvrp']
        
        cvrp_metrics = self._aggregate_results(cvrp_results) if cvrp_results else self._empty_metrics()
        twcvrp_metrics = self._aggregate_results(twcvrp_results) if twcvrp_results else self._empty_metrics()
        
        # Save results
        results = {
            'solver': solver_name,
            'overall': overall_metrics,
            'cvrp': cvrp_metrics,
            'twcvrp': twcvrp_metrics,
            'by_size': size_metrics,
            'detailed': all_results
        }
        
        self._save_results(results, solver_name)
        print(f"Evaluation completed. Results saved.")
        
        return results
    
    def _is_empty_dataset(self, data_dict: Dict) -> bool:
        """Check if dataset is empty or has no instances"""
        if 'locations' not in data_dict:
            return True
        
        if isinstance(data_dict['locations'], np.ndarray):
            return data_dict['locations'].shape[0] == 0
        elif isinstance(data_dict['locations'], list):
            return len(data_dict['locations']) == 0
        
        return True
    
    def _convert_to_dict(self, data) -> Dict:
        """Convert numpy archive to dictionary"""
        data_dict = {}
        for key in data.files if hasattr(data, 'files') else data:
            data_dict[key] = data[key]
        return data_dict
    
    def _limit_instances(self, data_dict: Dict, max_instances: int) -> Dict:
        """Limit number of instances for efficiency"""
        if 'locations' not in data_dict:
            return data_dict
        
        # Determine number of instances
        if isinstance(data_dict['locations'], np.ndarray):
            num_instances = min(max_instances, data_dict['locations'].shape[0])
        elif isinstance(data_dict['locations'], list):
            num_instances = min(max_instances, len(data_dict['locations']))
        else:
            return data_dict
        
        # Create limited dataset
        limited_data = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0 and value.shape[0] >= num_instances:
                limited_data[key] = value[:num_instances]
            elif isinstance(value, list) and len(value) >= num_instances:
                limited_data[key] = value[:num_instances]
            else:
                limited_data[key] = value
                
        return limited_data
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all instances"""
        if not all_results:
            return self._empty_metrics()
        
        metrics = {
            'total_cost': np.mean([r['metrics']['total_cost'] for r in all_results]),
            'waiting_time': np.mean([r['metrics'].get('waiting_time', 0) for r in all_results]),
            'cvr': np.mean([r['metrics']['cvr'] for r in all_results]),
            'feasibility': np.mean([r['metrics']['feasibility'] for r in all_results]),
            'runtime': np.mean([r['metrics']['runtime'] for r in all_results]),
            'robustness': np.mean([r['metrics']['robustness'] for r in all_results])
        }
        
        # Add time window violation metric if available
        tw_violations = [r['metrics'].get('time_window_violations', 0) for r in all_results]
        if any(tw > 0 for tw in tw_violations):
            metrics['time_window_violations'] = np.mean(tw_violations)
        
        return metrics
    
    def _aggregate_size_results(self, results_by_size: Dict) -> Dict:
        """Aggregate results by instance size"""
        size_metrics = {}
        for category, results_list in results_by_size.items():
            if results_list:
                size_metrics[category] = {
                    'feasibility': np.mean([r['feasibility'] for r in results_list]),
                    'cost': np.mean([r['total_cost'] for r in results_list]),
                    'cvr': np.mean([r['cvr'] for r in results_list]),
                    'runtime': np.mean([r['runtime'] for r in results_list])
                }
                
                # Add time window metrics if available
                if any('time_window_violations' in r for r in results_list):
                    size_metrics[category]['time_window_violations'] = np.mean(
                        [r.get('time_window_violations', 0) for r in results_list]
                    )
        return size_metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_cost': 0,
            'waiting_time': 0,
            'cvr': 0,
            'feasibility': 0,
            'runtime': 0,
            'robustness': 0,
            'time_window_violations': 0
        }
    
    def _save_results(self, results: Dict, solver_name: str):
        """Save results to JSON file"""
        filename = f"{solver_name.lower().replace(' ', '_')}_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_latex_tables(self, results: Dict):
        """Generate LaTeX tables with CVRP/TWCVRP separation"""
        solver_name = results['solver']
        
        print("\n" + "="*80)
        print(f"LATEX RESULTS FOR {solver_name.upper()}")
        print("="*80)
        
        # Table 1: Overall Performance Comparison
        self._print_overall_performance_latex(results)
        
        # Table 2: CVRP vs TWCVRP Performance
        self._print_problem_type_comparison_latex(results)
        
        # Table 3: Performance by Instance Size
        self._print_performance_by_size_latex(results)
        
        # Table 4: Detailed Metrics by Problem Type
        self._print_detailed_metrics_latex(results)
        
        # Table 5: Scalability Analysis
        self._print_scalability_analysis_latex(results)
    
    def _print_overall_performance_latex(self, results: Dict):
        """Generate LaTeX table for overall performance"""
        print("\n% Table 1: Overall Performance")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Overall Performance of " + results['solver'] + "}")
        print("\\label{tab:overall_performance}")
        print("\\begin{tabular}{lcrr}")
        print("\\hline")
        print("\\textbf{Metric} & \\textbf{Value} & \\textbf{Std} \\\\")
        print("\\hline")
        
        m = results['overall']
        print(f"Total Cost & {m['total_cost']:.1f} & {m.get('cost_std', 0):.1f} \\\\")
        print(f"CVR (\\%) & {m['cvr']:.1f} & {m.get('cvr_std', 0):.1f} \\\\")
        print(f"Feasibility & {m['feasibility']:.3f} & - \\\\")
        print(f"Runtime (ms) & {m['runtime']*1000:.1f} & {m.get('runtime_std', 0)*1000:.1f} \\\\")
        print(f"Robustness & {m['robustness']:.1f} & {m.get('robustness_std', 0):.1f} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def _print_problem_type_comparison_latex(self, results: Dict):
        """Generate LaTeX table comparing CVRP and TWCVRP"""
        print("\n% Table 2: CVRP vs TWCVRP Performance")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{CVRP vs TWCVRP Performance Comparison}")
        print("\\label{tab:problem_comparison}")
        print("\\begin{tabular}{lrrrr}")
        print("\\hline")
        print("\\textbf{Problem} & \\textbf{Cost} & \\textbf{CVR (\\%)} & \\textbf{Feasibility} & \\textbf{Runtime (ms)} \\\\")
        print("\\hline")
        
        cvrp = results.get('cvrp', self._empty_metrics())
        twcvrp = results.get('twcvrp', self._empty_metrics())
        
        print(f"CVRP & {cvrp['total_cost']:.1f} & {cvrp['cvr']:.1f} & {cvrp['feasibility']:.3f} & {cvrp['runtime']*1000:.1f} \\\\")
        print(f"TWCVRP & {twcvrp['total_cost']:.1f} & {twcvrp['cvr']:.1f} & {twcvrp['feasibility']:.3f} & {twcvrp['runtime']*1000:.1f} \\\\")
        
        # Show impact
        if cvrp['total_cost'] > 0:
            cost_increase = ((twcvrp['total_cost'] - cvrp['total_cost']) / cvrp['total_cost']) * 100
            print("\\hline")
            print(f"\\multicolumn{{5}}{{l}}{{\\textit{{Cost increase: {cost_increase:.1f}\\%}}}} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def _print_performance_by_size_latex(self, results: Dict):
        """Generate LaTeX table for performance by instance size"""
        print("\n% Table 3: Performance by Instance Size")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Performance by Instance Size}")
        print("\\label{tab:size_performance}")
        print("\\begin{tabular}{lrrrr}")
        print("\\hline")
        print("\\textbf{Size} & \\textbf{Feasibility} & \\textbf{Cost} & \\textbf{CVR (\\%)} & \\textbf{Runtime (ms)} \\\\")
        print("\\hline")
        
        for size in ['small', 'medium', 'large']:
            if size in results['by_size']:
                m = results['by_size'][size]
                print(f"{size.capitalize()} & {m['feasibility']:.3f} & {m['cost']:.1f} & {m['cvr']:.1f} & {m['runtime']*1000:.1f} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def _print_detailed_metrics_latex(self, results: Dict):
        """Generate detailed metrics table separated by problem type"""
        print("\n% Table 4: Detailed Metrics by Problem Type")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Detailed Metrics by Problem Type}")
        print("\\label{tab:detailed_metrics}")
        print("\\begin{tabular}{lrrr}")
        print("\\hline")
        print("\\textbf{Metric} & \\textbf{CVRP} & \\textbf{TWCVRP} & \\textbf{Difference} \\\\")
        print("\\hline")
        
        cvrp = results.get('cvrp', self._empty_metrics())
        twcvrp = results.get('twcvrp', self._empty_metrics())
        
        # Calculate differences
        cost_diff = twcvrp['total_cost'] - cvrp['total_cost']
        cvr_diff = twcvrp['cvr'] - cvrp['cvr']
        feas_diff = twcvrp['feasibility'] - cvrp['feasibility']
        runtime_diff = (twcvrp['runtime'] - cvrp['runtime']) * 1000
        
        print(f"Total Cost & {cvrp['total_cost']:.1f} & {twcvrp['total_cost']:.1f} & {cost_diff:+.1f} \\\\")
        print(f"CVR (\\%) & {cvrp['cvr']:.1f} & {twcvrp['cvr']:.1f} & {cvr_diff:+.1f} \\\\")
        print(f"Feasibility & {cvrp['feasibility']:.3f} & {twcvrp['feasibility']:.3f} & {feas_diff:+.3f} \\\\")
        print(f"Runtime (ms) & {cvrp['runtime']*1000:.1f} & {twcvrp['runtime']*1000:.1f} & {runtime_diff:+.1f} \\\\")
        
        # Add time window violations for TWCVRP
        if 'time_window_violations' in twcvrp and twcvrp['time_window_violations'] > 0:
            print("\\hline")
            print(f"TW Violations & - & {twcvrp['time_window_violations']:.2f} & - \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def _print_scalability_analysis_latex(self, results: Dict):
        """Generate scalability analysis table"""
        print("\n% Table 5: Scalability Analysis")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Scalability Analysis by Instance Size}")
        print("\\label{tab:scalability}")
        print("\\begin{tabular}{lrrr}")
        print("\\hline")
        print("\\textbf{Metric} & \\textbf{Small → Medium} & \\textbf{Medium → Large} & \\textbf{Overall Scaling} \\\\")
        print("\\hline")
        
        sizes = results.get('by_size', {})
        small = sizes.get('small', {})
        medium = sizes.get('medium', {})
        large = sizes.get('large', {})
        
        if small and medium and large:
            # Runtime scaling
            small_to_medium_rt = (medium['runtime'] / small['runtime']) if small['runtime'] > 0 else 1
            medium_to_large_rt = (large['runtime'] / medium['runtime']) if medium['runtime'] > 0 else 1
            overall_rt = (large['runtime'] / small['runtime']) if small['runtime'] > 0 else 1
            
            # Cost scaling
            small_to_medium_cost = (medium['cost'] / small['cost']) if small['cost'] > 0 else 1
            medium_to_large_cost = (large['cost'] / medium['cost']) if medium['cost'] > 0 else 1
            overall_cost = (large['cost'] / small['cost']) if small['cost'] > 0 else 1
            
            # Feasibility scaling
            small_to_medium_feas = (medium['feasibility'] / small['feasibility']) if small['feasibility'] > 0 else 0
            medium_to_large_feas = (large['feasibility'] / medium['feasibility']) if medium['feasibility'] > 0 else 0
            overall_feas = (large['feasibility'] / small['feasibility']) if small['feasibility'] > 0 else 0
            
            print(f"Runtime Factor & {small_to_medium_rt:.2f}x & {medium_to_large_rt:.2f}x & {overall_rt:.2f}x \\\\")
            print(f"Cost Factor & {small_to_medium_cost:.2f}x & {medium_to_large_cost:.2f}x & {overall_cost:.2f}x \\\\")
            print(f"Feasibility Factor & {small_to_medium_feas:.2f}x & {medium_to_large_feas:.2f}x & {overall_feas:.2f}x \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def generate_summary_insights(self, results: Dict):
        """Generate summary insights in text format"""
        print("\n" + "="*80)
        print(f"SUMMARY INSIGHTS FOR {results['solver'].upper()}")
        print("="*80)
        
        cvrp = results.get('cvrp', self._empty_metrics())
        twcvrp = results.get('twcvrp', self._empty_metrics())
        overall = results['overall']
        sizes = results.get('by_size', {})
        
        print(f"\n1. Overall Performance:")
        print(f"   - Average constraint violation rate: {overall['cvr']:.1f}%")
        print(f"   - Overall feasibility rate: {overall['feasibility']:.1%}")
        print(f"   - Average runtime: {overall['runtime']*1000:.1f} ms per instance")
        
        print(f"\n2. CVRP vs TWCVRP Impact:")
        if cvrp['total_cost'] > 0:
            cost_impact = ((twcvrp['total_cost'] - cvrp['total_cost']) / cvrp['total_cost']) * 100
            feas_impact = ((twcvrp['feasibility'] - cvrp['feasibility']) / cvrp['feasibility']) * 100 if cvrp['feasibility'] > 0 else 0
            runtime_impact = ((twcvrp['runtime'] - cvrp['runtime']) / cvrp['runtime']) * 100 if cvrp['runtime'] > 0 else 0
            
            print(f"   - Cost increase with time windows: {cost_impact:.1f}%")
            print(f"   - Feasibility change: {feas_impact:.1f}%")
            print(f"   - Runtime increase: {runtime_impact:.1f}%")
        
        print(f"\n3. Scalability Analysis:")
        for size, metrics in sizes.items():
            print(f"   - {size.capitalize()} instances ({self.size_categories[size]}-node problems):")
            print(f"     * Feasibility: {metrics['feasibility']:.1%}")
            print(f"     * Avg runtime: {metrics['runtime']*1000:.1f} ms")
            print(f"     * CVR: {metrics['cvr']:.1f}%")
        
        print(f"\n4. Time Window Compliance:")
        if 'time_window_violations' in twcvrp and twcvrp['time_window_violations'] > 0:
            print(f"   - Average TW violations per instance: {twcvrp['time_window_violations']:.2f}")
        else:
            print(f"   - No significant time window violations detected")
        
        print(f"\n5. Algorithm Efficiency:")
        if 'large' in sizes and 'small' in sizes:
            efficiency_small = sizes['small']['cost'] / sizes['small']['runtime'] if sizes['small']['runtime'] > 0 else 0
            efficiency_large = sizes['large']['cost'] / sizes['large']['runtime'] if sizes['large']['runtime'] > 0 else 0
            efficiency_ratio = efficiency_large / efficiency_small if efficiency_small > 0 else 0
            
            print(f"   - Small instance efficiency: {efficiency_small:.1f} cost per ms")
            print(f"   - Large instance efficiency: {efficiency_large:.1f} cost per ms")
            print(f"   - Efficiency scaling factor: {efficiency_ratio:.2f}x")