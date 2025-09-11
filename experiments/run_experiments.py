"""
Unified Experiment Runner for VPR SOTA Algorithms
Manages training and evaluation of multiple VPR algorithms with consistent setup
"""

import os
import sys
import argparse
import yaml
import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional
import pandas as pd


class VPRExperimentRunner:
    """Unified experiment runner for VPR algorithms"""
    
    def __init__(self, base_config_path: str, output_base_dir: str):
        self.base_config_path = base_config_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Algorithm configurations
        self.algorithms = {
            'netvlad': {
                'script': 'algorithms/netvlad/train_netvlad.py',
                'config': 'configs/netvlad_config.yaml'
            },
            'ap-gem': {
                'script': 'algorithms/ap-gem/train_apgem.py',
                'config': 'configs/apgem_config.yaml'
            },
            'delg': {
                'script': 'algorithms/delg/train_delg.py',
                'config': 'configs/delg_config.yaml'
            },
            'cosplace': {
                'script': 'algorithms/cosplace/train_cosplace.py',
                'config': 'configs/cosplace_config.yaml'
            },
            'eigenplaces': {
                'script': 'algorithms/eigenplaces/train_eigenplaces.py',
                'config': 'configs/eigenplaces_config.yaml'
            }
        }
        
        # Results storage
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = self.output_base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"experiment_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def load_base_config(self) -> Dict:
        """Load base configuration"""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_algorithm_config(self, algorithm: str, base_config: Dict) -> str:
        """Create algorithm-specific configuration"""
        
        # Load algorithm template
        template_config_path = self.algorithms[algorithm]['config']
        with open(template_config_path, 'r') as f:
            template_config = yaml.safe_load(f)
        
        # Update with base config values
        algorithm_config = template_config.copy()
        
        # Override common settings from base config
        for key in ['train_csv', 'test_csv', 'train_base_path', 'test_base_path', 
                   'camera_filter', 'positive_threshold', 'negative_threshold']:
            if key in base_config:
                algorithm_config[key] = base_config[key]
        
        # Set algorithm-specific output directory
        algorithm_config['output_dir'] = str(self.output_base_dir / algorithm)
        
        # Override any algorithm-specific settings from base config
        if 'algorithm_settings' in base_config and algorithm in base_config['algorithm_settings']:
            algorithm_config.update(base_config['algorithm_settings'][algorithm])
        
        # Save algorithm config
        config_dir = self.output_base_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f"{algorithm}_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(algorithm_config, f, default_flow_style=False)
        
        return str(config_path)
    
    def run_algorithm(self, algorithm: str, config_path: str) -> Dict:
        """Run a single algorithm"""
        self.logger.info(f"Starting training for {algorithm}...")
        
        script_path = self.algorithms[algorithm]['script']
        
        # Create output directory
        output_dir = self.output_base_dir / algorithm
        output_dir.mkdir(exist_ok=True)
        
        # Run training script
        cmd = [
            sys.executable, script_path,
            '--config', config_path
        ]
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the training script
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),  # Run from project root
                capture_output=True,
                text=True,
                timeout=3600 * 6  # 6 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully completed training for {algorithm}")
                
                # Try to load results from the output directory
                results = self._load_algorithm_results(algorithm, output_dir)
                
                return {
                    'status': 'success',
                    'results': results,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                self.logger.error(f"Training failed for {algorithm}: {result.stderr}")
                return {
                    'status': 'failed',
                    'error': result.stderr,
                    'stdout': result.stdout
                }
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"Training timed out for {algorithm}")
            return {
                'status': 'timeout',
                'error': 'Training timed out after 6 hours'
            }
        except Exception as e:
            self.logger.error(f"Error running {algorithm}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _load_algorithm_results(self, algorithm: str, output_dir: Path) -> Dict:
        """Load results from algorithm output directory"""
        results = {}
        
        # Look for best model results
        best_model_path = output_dir / 'best_model.pth'
        if best_model_path.exists():
            try:
                import torch
                checkpoint = torch.load(best_model_path, map_location='cpu')
                if 'results' in checkpoint:
                    results = checkpoint['results']
            except Exception as e:
                self.logger.warning(f"Could not load results from {best_model_path}: {e}")
        
        # Look for evaluation results file
        eval_results_path = output_dir / 'evaluation_results.json'
        if eval_results_path.exists():
            try:
                with open(eval_results_path, 'r') as f:
                    results.update(json.load(f))
            except Exception as e:
                self.logger.warning(f"Could not load evaluation results: {e}")
        
        return results
    
    def run_experiments(self, algorithms: Optional[List[str]] = None) -> Dict:
        """Run experiments for specified algorithms"""
        
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        # Load base configuration
        base_config = self.load_base_config()
        
        self.logger.info(f"Starting experiments for algorithms: {algorithms}")
        self.logger.info(f"Base configuration: {base_config}")
        
        results = {}
        
        for algorithm in algorithms:
            if algorithm not in self.algorithms:
                self.logger.error(f"Unknown algorithm: {algorithm}")
                continue
            
            self.logger.info(f"Running experiment for {algorithm}")
            
            # Create algorithm-specific config
            config_path = self.create_algorithm_config(algorithm, base_config)
            
            # Run algorithm
            result = self.run_algorithm(algorithm, config_path)
            results[algorithm] = result
            
            # Save intermediate results
            self._save_results(results)
        
        self.logger.info("All experiments completed!")
        return results
    
    def _save_results(self, results: Dict):
        """Save experiment results"""
        results_file = self.output_base_dir / 'experiment_results.json'
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for alg, result in results.items():
            serializable_results[alg] = {
                'status': result['status'],
                'results': result.get('results', {}),
                'error': result.get('error', ''),
                'timestamp': datetime.now().isoformat()
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_comparison_report(self, results: Dict) -> str:
        """Generate a comparison report from experiment results"""
        
        report_lines = []
        report_lines.append("VPR SOTA Algorithms Comparison Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("Algorithm Performance Summary")
        report_lines.append("-" * 30)
        
        comparison_data = []
        
        for algorithm, result in results.items():
            if result['status'] == 'success' and 'results' in result:
                res = result['results']
                if 'recall_at_k' in res:
                    comparison_data.append({
                        'Algorithm': algorithm.upper(),
                        'R@1': f"{res['recall_at_k'].get(1, 0.0):.3f}",
                        'R@5': f"{res['recall_at_k'].get(5, 0.0):.3f}",
                        'R@10': f"{res['recall_at_k'].get(10, 0.0):.3f}",
                        'R@20': f"{res['recall_at_k'].get(20, 0.0):.3f}",
                        'mAP': f"{res.get('mean_average_precision', 0.0):.3f}",
                        'Status': 'Success'
                    })
                else:
                    comparison_data.append({
                        'Algorithm': algorithm.upper(),
                        'R@1': 'N/A',
                        'R@5': 'N/A',
                        'R@10': 'N/A',
                        'R@20': 'N/A',
                        'mAP': 'N/A',
                        'Status': 'Success (No metrics)'
                    })
            else:
                comparison_data.append({
                    'Algorithm': algorithm.upper(),
                    'R@1': 'N/A',
                    'R@5': 'N/A',
                    'R@10': 'N/A',
                    'R@20': 'N/A',
                    'mAP': 'N/A',
                    'Status': result['status'].title()
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report_lines.append(df.to_string(index=False))
        else:
            report_lines.append("No successful results to compare.")
        
        report_lines.append("")
        report_lines.append("Detailed Results")
        report_lines.append("-" * 20)
        
        for algorithm, result in results.items():
            report_lines.append(f"\n{algorithm.upper()}:")
            report_lines.append(f"  Status: {result['status']}")
            
            if result['status'] == 'success' and 'results' in result:
                res = result['results']
                report_lines.append(f"  Total queries: {res.get('total_queries', 'N/A')}")
                report_lines.append(f"  Queries with matches: {res.get('queries_with_matches', 'N/A')}")
                report_lines.append(f"  Distance threshold: {res.get('distance_threshold', 'N/A')}m")
                
                if 'recall_at_k' in res:
                    report_lines.append("  Recall@K:")
                    for k, recall in res['recall_at_k'].items():
                        report_lines.append(f"    R@{k}: {recall:.3f}")
                
                if 'precision_at_k' in res:
                    report_lines.append("  Precision@K:")
                    for k, precision in res['precision_at_k'].items():
                        report_lines.append(f"    P@{k}: {precision:.3f}")
                
                if 'mean_average_precision' in res:
                    report_lines.append(f"  mAP: {res['mean_average_precision']:.3f}")
            
            elif result['status'] == 'failed':
                report_lines.append(f"  Error: {result.get('error', 'Unknown error')}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_base_dir / 'comparison_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return report_content


def main():
    parser = argparse.ArgumentParser(description='Run VPR SOTA Experiments')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to base experiment configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Base output directory for all experiments')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       help='Algorithms to run (default: all)')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = VPRExperimentRunner(args.config, args.output_dir)
    
    # Run experiments
    results = runner.run_experiments(args.algorithms)
    
    # Generate comparison report
    report = runner.generate_comparison_report(results)
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    print(report)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Detailed logs available in: {args.output_dir}/logs/")


if __name__ == "__main__":
    main()
