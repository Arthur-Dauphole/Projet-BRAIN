"""
ARC Geometric Detection Test Runner
====================================

Automated test suite for the geometric detection system.
Loads test grids from JSON and validates detection results.

Usage:
    python test_runner.py [--visualize] [--verbose] [--filter TAG]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from dataclasses import dataclass
from datetime import datetime

# Import the geometric detection system
# Note: Make sure arc_geometric_detection.py is in the same directory
from arc_geometric_detection import (
    GeometricDetectionEngine,
    GeometricVisualizer,
    GeometricShape
)


# ============================================================================
# TEST RESULT TRACKING
# ============================================================================

@dataclass
class TestResult:
    """Store results of a single test case."""
    test_id: str
    test_name: str
    passed: bool
    expected_rectangles: int
    detected_rectangles: int
    expected_lines: int
    detected_lines: int
    error_message: str = ""
    
    def to_dict(self):
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'passed': self.passed,
            'expected': {
                'rectangles': self.expected_rectangles,
                'lines': self.expected_lines
            },
            'detected': {
                'rectangles': self.detected_rectangles,
                'lines': self.detected_lines
            },
            'error': self.error_message
        }


class TestSuite:
    """
    Main test suite runner for geometric detection.
    
    Loads test cases from JSON, runs detection, validates results,
    and generates comprehensive reports.
    """
    
    def __init__(self, json_path: str = "arc_test_grids.json"):
        """
        Initialize test suite.
        
        Args:
            json_path: Path to JSON file containing test grids
        """
        self.json_path = json_path
        self.test_data = None
        self.results: List[TestResult] = []
        self.engine = GeometricDetectionEngine(background_color=0)
        
    def load_tests(self) -> bool:
        """
        Load test cases from JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Construct path relative to the script's location
            script_dir = Path(__file__).parent
            full_json_path = script_dir / self.json_path
            with open(full_json_path, 'r') as f:
                self.test_data = json.load(f)
            
            print(f"✓ Loaded {len(self.test_data['test_grids'])} test cases from {full_json_path}")
            return True
        
        except FileNotFoundError:
            print(f"✗ Error: Could not find file '{self.json_path}' in the script directory.")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON format - {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading tests: {e}")
            return False
    
    def run_single_test(self, test_case: Dict, verbose: bool = False) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: Dictionary containing test case data
            verbose: Whether to print detailed output
        
        Returns:
            TestResult object
        """
        test_id = test_case['id']
        test_name = test_case['name']
        grid = np.array(test_case['grid'])
        expected = test_case['expected_shapes']
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing: {test_name} ({test_id})")
            print(f"Description: {test_case['description']}")
            print(f"Tags: {', '.join(test_case['tags'])}")
            print(f"Grid size: {grid.shape[1]}x{grid.shape[0]}")
        
        try:
            # Run detection
            analysis = self.engine.analyze_grid(grid, verbose=False)
            
            detected_rectangles = len(analysis['detected_shapes']['rectangles'])
            detected_lines = len(analysis['detected_shapes']['lines'])
            
            expected_rectangles = expected['rectangles']
            expected_lines = expected['lines']
            
            # Check if results match expectations
            rectangles_match = detected_rectangles == expected_rectangles
            lines_match = detected_lines == expected_lines
            passed = rectangles_match and lines_match
            
            error_msg = ""
            if not passed:
                errors = []
                if not rectangles_match:
                    errors.append(
                        f"Rectangles: expected {expected_rectangles}, got {detected_rectangles}"
                    )
                if not lines_match:
                    errors.append(
                        f"Lines: expected {expected_lines}, got {detected_lines}"
                    )
                error_msg = "; ".join(errors)
            
            if verbose:
                print(f"\nExpected: {expected_rectangles} rectangles, {expected_lines} lines")
                print(f"Detected: {detected_rectangles} rectangles, {detected_lines} lines")
                
                if passed:
                    print("✓ PASSED")
                else:
                    print(f"✗ FAILED: {error_msg}")
            
            return TestResult(
                test_id=test_id,
                test_name=test_name,
                passed=passed,
                expected_rectangles=expected_rectangles,
                detected_rectangles=detected_rectangles,
                expected_lines=expected_lines,
                detected_lines=detected_lines,
                error_message=error_msg
            )
        
        except Exception as e:
            if verbose:
                print(f"✗ EXCEPTION: {str(e)}")
            
            return TestResult(
                test_id=test_id,
                test_name=test_name,
                passed=False,
                expected_rectangles=expected['rectangles'],
                detected_rectangles=0,
                expected_lines=expected['lines'],
                detected_lines=0,
                error_message=f"Exception: {str(e)}"
            )
    
    def run_all_tests(self, verbose: bool = False, filter_tag: str = None) -> List[TestResult]:
        """
        Run all test cases.
        
        Args:
            verbose: Whether to print detailed output for each test
            filter_tag: Optional tag to filter tests (e.g., 'rectangle', 'line')
        
        Returns:
            List of TestResult objects
        """
        if self.test_data is None:
            print("✗ No test data loaded. Call load_tests() first.")
            return []
        
        test_grids = self.test_data['test_grids']
        
        # Filter tests if tag is specified
        if filter_tag:
            test_grids = [t for t in test_grids if filter_tag in t.get('tags', [])]
            print(f"\nFiltering tests with tag: '{filter_tag}'")
            print(f"Found {len(test_grids)} matching tests\n")
        
        self.results = []
        
        print(f"\n{'='*70}")
        print(f"RUNNING TEST SUITE")
        print(f"{'='*70}\n")
        
        for i, test_case in enumerate(test_grids, 1):
            if not verbose:
                print(f"[{i}/{len(test_grids)}] {test_case['name']}...", end=" ")
            
            result = self.run_single_test(test_case, verbose=verbose)
            self.results.append(result)
            
            if not verbose:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(status)
                if not result.passed:
                    print(f"    └─ {result.error_message}")
        
        return self.results
    
    def generate_report(self) -> Dict:
        """
        Generate summary report of test results.
        
        Returns:
            Dictionary containing report data
        """
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Categorize failures
        rectangle_failures = []
        line_failures = []
        exception_failures = []
        
        for result in self.results:
            if not result.passed:
                if "Exception" in result.error_message:
                    exception_failures.append(result)
                elif "Rectangle" in result.error_message:
                    rectangle_failures.append(result)
                elif "Line" in result.error_message:
                    line_failures.append(result)
        
        report = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'pass_rate': pass_rate,
            'failures_by_category': {
                'rectangles': len(rectangle_failures),
                'lines': len(line_failures),
                'exceptions': len(exception_failures)
            },
            'failed_tests': [r.to_dict() for r in self.results if not r.passed]
        }
        
        return report
    
    def print_report(self):
        """Print formatted test report to console."""
        report = self.generate_report()
        
        if not report:
            print("No test results to report.")
            return
        
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Total tests run:     {report['total_tests']}")
        print(f"Tests passed:        {report['passed']} ✓")
        print(f"Tests failed:        {report['failed']} ✗")
        print(f"Pass rate:           {report['pass_rate']:.1f}%")
        
        if report['failed'] > 0:
            print(f"\n{'='*70}")
            print(f"FAILURE BREAKDOWN")
            print(f"{'='*70}\n")
            
            failures = report['failures_by_category']
            print(f"Rectangle detection failures: {failures['rectangles']}")
            print(f"Line detection failures:      {failures['lines']}")
            print(f"Exception failures:           {failures['exceptions']}")
            
            print(f"\n{'='*70}")
            print(f"FAILED TESTS DETAILS")
            print(f"{'='*70}\n")
            
            for i, failure in enumerate(report['failed_tests'], 1):
                print(f"{i}. {failure['test_name']} ({failure['test_id']})")
                print(f"   Error: {failure['error']}")
                print()
        
        # Overall status
        print(f"{'='*70}")
        if report['pass_rate'] == 100:
            print("🎉 ALL TESTS PASSED!")
        elif report['pass_rate'] >= 80:
            print("⚠️  MOSTLY PASSING (some failures)")
        else:
            print("❌ MULTIPLE FAILURES DETECTED")
        print(f"{'='*70}\n")
    
    def visualize_test(self, test_id: str, save_path: str = None):
        """
        Visualize a specific test case with detected shapes.
        
        Args:
            test_id: ID of the test case to visualize
            save_path: Optional path to save the visualization
        """
        if self.test_data is None:
            print("✗ No test data loaded.")
            return
        
        # Find test case
        test_case = None
        for tc in self.test_data['test_grids']:
            if tc['id'] == test_id:
                test_case = tc
                break
        
        if test_case is None:
            print(f"✗ Test case '{test_id}' not found.")
            return
        
        # Run detection
        grid = np.array(test_case['grid'])
        analysis = self.engine.analyze_grid(grid, verbose=False)
        
        # Collect all shapes
        all_shapes = []
        for shapes in analysis['detected_shapes'].values():
            all_shapes.extend(shapes)
        
        # Create visualization
        title = f"{test_case['name']} - Detected: {len(all_shapes)} shapes"
        fig, ax = GeometricVisualizer.plot_shapes(grid, all_shapes, title=title)
        
        # Add test info as text
        info_text = f"Expected: {test_case['expected_shapes']}\n"
        info_text += f"Tags: {', '.join(test_case['tags'])}"
        fig.text(0.02, 0.02, info_text, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            # Close the plot to free up memory
            plt.close(fig)
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()
    
    def visualize_all_failed_tests(self, output_dir: str):
        """
        Generate visualizations for all failed tests.
        
        Args:
            output_dir: Directory to save visualizations
        """
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        failed_results = [r for r in self.results if not r.passed]
        
        if not failed_results:
            print("✓ No failed tests to visualize!")
            return
        
        print(f"\nGenerating visualizations for {len(failed_results)} failed tests...")
        
        for result in failed_results:
            output_path = Path(output_dir) / f"{result.test_id}.png"
            self.visualize_test(result.test_id, save_path=str(output_path))
        
        print(f"✓ Visualizations saved to {output_dir}/")
    
    def export_results(self, output_path: str):
        """
        Export test results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Results exported to {output_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Run geometric detection tests on ARC-like grids"
    )
    
    parser.add_argument(
        '--json', '-j',
        default='arc_test_grids.json',
        help='Path to test grids JSON file (default: arc_test_grids.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output for each test'
    )
    
    parser.add_argument(
        '--filter', '-f',
        type=str,
        default=None,
        help='Filter tests by tag (e.g., "rectangle", "line", "symmetry")'
    )
    
    parser.add_argument(
        '--visualize', '-viz',
        action='store_true',
        help='Generate visualizations for failed tests'
    )
    
    parser.add_argument(
        '--visualize-all', '-viz-all',
        action='store_true',
        help='Generate visualizations for ALL tests (not just failures)'
    )
    
    parser.add_argument(
        '--test-id', '-t',
        type=str,
        default=None,
        help='Visualize a specific test by ID'
    )
    
    parser.add_argument(
        '--export', '-e',
        type=str,
        default=None,
        help='Export results to JSON file (specify path)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='test_failures',
        help='Base output directory for visualizations (default: test_failures)'
    )
    
    args = parser.parse_args()
    
    # Initialize test suite
    suite = TestSuite(json_path=args.json)
    
    # Load tests
    if not suite.load_tests():
        return 1
    
    # Handle single test visualization
    if args.test_id:
        suite.visualize_test(args.test_id)
        return 0
    
    # Run all tests
    suite.run_all_tests(verbose=args.verbose, filter_tag=args.filter)
    
    # Print report
    suite.print_report()
    
    # Export results if requested
    if args.export:
        suite.export_results(args.export)
    
    # Generate visualizations if requested
    if args.visualize or args.visualize_all:
        # --- MODIFICATION POUR LE CHEMIN RELATIF ---
        # Get the directory where the script is located
        script_dir = Path(__file__).parent
        
        # Create a timestamped directory name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Construct the full path for the output directory inside the script's folder
        timestamped_dir = script_dir / f"{args.output_dir}_{timestamp}"
        
        if args.visualize:
            suite.visualize_all_failed_tests(output_dir=str(timestamped_dir))
        
        if args.visualize_all:
            timestamped_dir.mkdir(exist_ok=True, parents=True)
            print(f"\nGenerating visualizations for all {len(suite.results)} tests...")
            for result in suite.results:
                output_path = timestamped_dir / f"{result.test_id}.png"
                suite.visualize_test(result.test_id, save_path=str(output_path))
            print(f"✓ All visualizations saved to {timestamped_dir}/")
    
    # Return exit code based on test results
    report = suite.generate_report()
    return 0 if report.get('pass_rate', 0) == 100 else 1


# ============================================================================
# SIMPLE USAGE (if run directly without command line arguments)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # If no command line arguments, run in simple mode
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("ARC GEOMETRIC DETECTION TEST SUITE - SIMPLE MODE")
        print("="*70 + "\n")
        print("Running all tests with default settings...")
        print("(Use --help to see all available options)\n")
        
        # Create and run test suite
        suite = TestSuite()
        
        if suite.load_tests():
            suite.run_all_tests(verbose=False)
            suite.print_report()
            
            # Ask if user wants to see visualizations
            report = suite.generate_report()
            if report.get('failed', 0) > 0:
                print("\nWould you like to generate visualizations of failed tests?")
                response = input("(y/n): ").strip().lower()
                if response == 'y':
                    # --- MODIFICATION POUR LE CHEMIN RELATIF ---
                    # Get the directory where the script is located
                    script_dir = Path(__file__).parent
                    
                    # Create a timestamped directory for simple mode as well
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_dir = script_dir / f"test_failures_{timestamp}"
                    suite.visualize_all_failed_tests(output_dir=str(output_dir))
        
        sys.exit(0)
    
    # Otherwise, use command line argument parser
    sys.exit(main())