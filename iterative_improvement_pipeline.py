#!/usr/bin/env python3
"""
Automated Iterative Improvement Pipeline
======================================

This script implements a fully automated pipeline for iterative model improvement:

1. Process PDFs with current model
2. Apply intelligent filtering to reduce false positives
3. Generate predictions for manual review
4. Analyze filtering feedback and model performance
5. Retrain model with enhanced data and adjusted parameters
6. Repeat the cycle

Features:
✅ Fully automated iterative improvement
✅ Intelligent filtering with feedback analysis
✅ Performance tracking across iterations
✅ Automatic parameter adjustment
✅ Manual review integration
✅ Convergence detection
✅ Comprehensive reporting

Usage:
    python iterative_improvement_pipeline.py

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import json
import logging
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IterativeImprovementPipeline:
    """Automated pipeline for iterative model improvement"""
    
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Load configuration
        self.config = self.load_config()
        
        # Core directories
        self.input_dir = os.path.join(self.base_dir, self.config['directories']['input'])
        self.unprocessed_pdfs_dir = os.path.join(self.base_dir, self.config['directories']['unprocessed_pdfs'])
        self.predictions_dir = os.path.join(self.base_dir, self.config['directories']['predictions'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        self.reviewed_dir = os.path.join(self.base_dir, self.config['directories']['reviewed'])
        
        # Pipeline state
        self.current_iteration = 0
        self.performance_history = []
        self.convergence_achieved = False
        
        # Create pipeline results directory
        self.pipeline_results_dir = os.path.join(self.base_dir, "pipeline_results")
        os.makedirs(self.pipeline_results_dir, exist_ok=True)
        
        logger.info("🔄 Iterative Improvement Pipeline initialized!")
        logger.info(f"🎯 Max iterations: {max_iterations}")
        logger.info(f"🎯 Convergence threshold: {convergence_threshold}")
        logger.info(f"📁 Pipeline results: {self.pipeline_results_dir}")
    
    def load_config(self):
        """Load configuration"""
        config_path = os.path.join(self.base_dir, 'config_main.json')
        
        if not os.path.exists(config_path):
            logger.error(f"❌ Config file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise
    
    def check_dependencies(self):
        """Check if all required scripts and dependencies are available"""
        logger.info("🔍 Checking pipeline dependencies...")
        
        required_scripts = [
            'process_pdfs.py',
            'generate_json_output.py',
            'intelligent_filter.py',
            'enhanced_retraining.py'
        ]
        
        missing_scripts = []
        for script in required_scripts:
            script_path = os.path.join(self.base_dir, script)
            if not os.path.exists(script_path):
                missing_scripts.append(script)
        
        if missing_scripts:
            logger.error(f"❌ Missing required scripts: {missing_scripts}")
            return False
        
        logger.info("✅ All required scripts found")
        return True
    
    def process_pdfs_with_current_model(self) -> Dict:
        """Process PDFs with the current model"""
        logger.info(f"📄 [Iteration {self.current_iteration}] Processing PDFs with current model...")
        
        try:
            # Import and run PDF processing
            sys.path.insert(0, self.base_dir)
            from generate_json_output import JSONOutputGenerator
            
            # Initialize generator
            generator = JSONOutputGenerator()
            
            # Load latest model
            generator.load_model("latest")
            
            # Process PDFs from unprocessed_pdfs directory
            result = generator.generate_json_output(source_dir="unprocessed_pdfs")
            
            logger.info(f"✅ PDF processing completed for iteration {self.current_iteration}")
            return {"status": "success", "result": result}
            
        except Exception as e:
            logger.error(f"❌ Error processing PDFs: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_filtering_performance(self) -> Dict:
        """Analyze filtering performance from the latest reports"""
        logger.info(f"📊 [Iteration {self.current_iteration}] Analyzing filtering performance...")
        
        try:
            # Find latest filtering reports
            filtering_reports = sorted(
                Path(self.base_dir).glob("filtering_report_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not filtering_reports:
                logger.warning("⚠️  No filtering reports found")
                return {"status": "no_reports"}
            
            # Analyze the latest report
            latest_report_path = filtering_reports[0]
            with open(latest_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # Extract key metrics
            filtering_summary = report.get('filtering_summary', {})
            performance_metrics = {
                'original_predictions': filtering_summary.get('original_predictions', 0),
                'final_predictions': filtering_summary.get('final_predictions', 0),
                'reduction_rate': filtering_summary.get('reduction_rate', 0),
                'total_blocks': filtering_summary.get('total_blocks', 0),
                'iteration': self.current_iteration,
                'report_path': str(latest_report_path)
            }
            
            # Calculate false positive rate estimate
            if performance_metrics['original_predictions'] > 0:
                performance_metrics['estimated_fp_rate'] = performance_metrics['reduction_rate']
            else:
                performance_metrics['estimated_fp_rate'] = 0
            
            logger.info(f"📈 Filtering Performance Analysis:")
            logger.info(f"   🔢 Original predictions: {performance_metrics['original_predictions']}")
            logger.info(f"   ✅ Final predictions: {performance_metrics['final_predictions']}")
            logger.info(f"   📉 Reduction rate: {performance_metrics['reduction_rate']:.1f}%")
            logger.info(f"   📊 Estimated FP rate: {performance_metrics['estimated_fp_rate']:.1f}%")
            
            return {"status": "success", "metrics": performance_metrics}
            
        except Exception as e:
            logger.error(f"❌ Error analyzing filtering performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def retrain_model_with_feedback(self) -> Dict:
        """Retrain model using enhanced feedback"""
        logger.info(f"🎯 [Iteration {self.current_iteration}] Retraining model with feedback...")
        
        try:
            # Import and run enhanced retraining
            sys.path.insert(0, self.base_dir)
            from enhanced_retraining import EnhancedModelRetrainer
            
            # Initialize retrainer
            retrainer = EnhancedModelRetrainer()
            
            # Perform enhanced retraining
            model_version = retrainer.retrain_with_enhanced_feedback()
            
            logger.info(f"✅ Model retraining completed: {model_version}")
            return {"status": "success", "model_version": model_version}
            
        except Exception as e:
            logger.error(f"❌ Error retraining model: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_convergence(self) -> bool:
        """Check if the model has converged"""
        if len(self.performance_history) < 2:
            return False
        
        # Get last two performance measurements
        current = self.performance_history[-1]
        previous = self.performance_history[-2]
        
        # Check for convergence based on reduction rate improvement
        current_fp_rate = current.get('estimated_fp_rate', 100)
        previous_fp_rate = previous.get('estimated_fp_rate', 100)
        
        improvement = previous_fp_rate - current_fp_rate
        
        logger.info(f"🔍 Convergence Check:")
        logger.info(f"   📈 Previous FP rate: {previous_fp_rate:.1f}%")
        logger.info(f"   📈 Current FP rate: {current_fp_rate:.1f}%")
        logger.info(f"   📈 Improvement: {improvement:.1f}%")
        logger.info(f"   🎯 Threshold: {self.convergence_threshold*100:.1f}%")
        
        # Convergence achieved if improvement is below threshold
        if abs(improvement) < self.convergence_threshold * 100:
            logger.info("✅ Convergence achieved!")
            return True
        
        logger.info("⏳ Convergence not yet achieved, continuing...")
        return False
    
    def generate_iteration_report(self, iteration_results: Dict):
        """Generate comprehensive report for the current iteration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.pipeline_results_dir, f"iteration_{self.current_iteration}_report_{timestamp}.json")
        
        report = {
            "iteration": self.current_iteration,
            "timestamp": timestamp,
            "pipeline_config": {
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold
            },
            "iteration_results": iteration_results,
            "performance_history": self.performance_history,
            "convergence_status": {
                "achieved": self.convergence_achieved,
                "remaining_iterations": self.max_iterations - self.current_iteration
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Iteration report saved: {report_path}")
        return report_path
    
    def run_iteration(self) -> Dict:
        """Run a single iteration of the improvement pipeline"""
        logger.info(f"🚀 Starting iteration {self.current_iteration}")
        
        iteration_results = {
            "iteration": self.current_iteration,
            "start_time": datetime.now().isoformat(),
            "steps": {}
        }
        
        # Step 1: Process PDFs with current model
        logger.info("📝 Step 1: Processing PDFs with current model...")
        pdf_result = self.process_pdfs_with_current_model()
        iteration_results["steps"]["pdf_processing"] = pdf_result
        
        if pdf_result["status"] != "success":
            logger.error("❌ PDF processing failed, aborting iteration")
            return iteration_results
        
        # Step 2: Analyze filtering performance
        logger.info("📊 Step 2: Analyzing filtering performance...")
        analysis_result = self.analyze_filtering_performance()
        iteration_results["steps"]["performance_analysis"] = analysis_result
        
        if analysis_result["status"] == "success":
            # Store performance metrics
            self.performance_history.append(analysis_result["metrics"])
        
        # Step 3: Check convergence (if we have enough data)
        if len(self.performance_history) >= 2:
            logger.info("🔍 Step 3: Checking convergence...")
            self.convergence_achieved = self.check_convergence()
            iteration_results["convergence_achieved"] = self.convergence_achieved
            
            if self.convergence_achieved:
                logger.info("🎯 Convergence achieved! Stopping iterations.")
                iteration_results["end_time"] = datetime.now().isoformat()
                return iteration_results
        
        # Step 4: Retrain model with feedback (if not the last iteration)
        if self.current_iteration < self.max_iterations:
            logger.info("🎯 Step 4: Retraining model with feedback...")
            retrain_result = self.retrain_model_with_feedback()
            iteration_results["steps"]["model_retraining"] = retrain_result
            
            if retrain_result["status"] != "success":
                logger.error("❌ Model retraining failed")
        
        iteration_results["end_time"] = datetime.now().isoformat()
        return iteration_results
    
    def run_pipeline(self):
        """Run the complete iterative improvement pipeline"""
        logger.info("🚀 STARTING ITERATIVE IMPROVEMENT PIPELINE")
        logger.info("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed, aborting pipeline")
            return
        
        pipeline_start_time = datetime.now()
        pipeline_results = {
            "pipeline_start": pipeline_start_time.isoformat(),
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "iterations": []
        }
        
        try:
            # Run iterations
            for iteration in range(1, self.max_iterations + 1):
                self.current_iteration = iteration
                
                logger.info(f"\n🔄 ===== ITERATION {iteration}/{self.max_iterations} =====")
                
                # Run single iteration
                iteration_results = self.run_iteration()
                pipeline_results["iterations"].append(iteration_results)
                
                # Generate iteration report
                report_path = self.generate_iteration_report(iteration_results)
                
                # Check if convergence was achieved
                if self.convergence_achieved:
                    logger.info(f"🎯 Pipeline completed early due to convergence at iteration {iteration}")
                    break
                
                # Wait between iterations to allow for manual review if needed
                if iteration < self.max_iterations:
                    logger.info("⏳ Waiting 30 seconds before next iteration...")
                    time.sleep(30)
            
            # Pipeline completion
            pipeline_end_time = datetime.now()
            pipeline_results["pipeline_end"] = pipeline_end_time.isoformat()
            pipeline_results["total_duration"] = str(pipeline_end_time - pipeline_start_time)
            pipeline_results["convergence_achieved"] = self.convergence_achieved
            pipeline_results["final_iteration"] = self.current_iteration
            
            # Generate final pipeline report
            final_report_path = os.path.join(
                self.pipeline_results_dir,
                f"pipeline_final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(final_report_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
            
            logger.info("\n" + "=" * 60)
            logger.info("🏁 ITERATIVE IMPROVEMENT PIPELINE COMPLETED")
            logger.info("=" * 60)
            logger.info(f"📊 Total iterations: {self.current_iteration}")
            logger.info(f"🎯 Convergence achieved: {self.convergence_achieved}")
            logger.info(f"⏱️  Total duration: {pipeline_results['total_duration']}")
            logger.info(f"📋 Final report: {final_report_path}")
            
            if self.performance_history:
                final_performance = self.performance_history[-1]
                logger.info(f"📈 Final FP reduction rate: {final_performance.get('reduction_rate', 0):.1f}%")
                logger.info(f"📈 Final estimated FP rate: {final_performance.get('estimated_fp_rate', 0):.1f}%")
            
            return pipeline_results
            
        except KeyboardInterrupt:
            logger.info("\n⏹️  Pipeline interrupted by user")
            return pipeline_results
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {e}")
            raise


def main():
    """Main function"""
    print("🔄 AUTOMATED ITERATIVE IMPROVEMENT PIPELINE")
    print("=" * 60)
    print("🎯 This pipeline will automatically:")
    print("   1. Process PDFs with the current model")
    print("   2. Apply intelligent filtering to reduce false positives")
    print("   3. Analyze filtering performance and feedback")
    print("   4. Retrain the model with enhanced data")
    print("   5. Repeat until convergence or max iterations")
    print()
    print("⚠️  IMPORTANT:")
    print("   • Make sure you have PDFs in the unprocessed_pdfs/ folder")
    print("   • The pipeline will run automatically with minimal user intervention")
    print("   • You can interrupt with Ctrl+C if needed")
    print("   • All results will be saved in pipeline_results/")
    print()
    
    # Get user confirmation
    response = input("🚀 Start the automated pipeline? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("⏹️  Pipeline cancelled by user")
        return
    
    # Get pipeline parameters
    try:
        max_iterations = int(input("📝 Maximum iterations (default: 5): ") or "5")
        convergence_threshold = float(input("📝 Convergence threshold % (default: 5.0): ") or "5.0") / 100
    except ValueError:
        print("❌ Invalid input, using defaults")
        max_iterations = 5
        convergence_threshold = 0.05
    
    try:
        # Initialize and run pipeline
        pipeline = IterativeImprovementPipeline(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )
        
        # Run the pipeline
        results = pipeline.run_pipeline()
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Check pipeline_results/ for detailed reports")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    main()
