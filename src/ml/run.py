"""Main script to run the ML workflow."""

import argparse
from pathlib import Path
from core.workflow.workflow_manager import WorkflowManager
from core.utils.logging import setup_logging

logger = setup_logging(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ML workflow")
    
    # Configuration paths
    parser.add_argument("--model-config", type=Path, required=True,
                       help="Path to model configuration file")
    parser.add_argument("--training-config", type=Path, required=True,
                       help="Path to training configuration file")
    parser.add_argument("--data-config", type=Path, required=True,
                       help="Path to data configuration file")
    parser.add_argument("--task-config", type=Path,
                       help="Path to task-specific configuration file")
    
    # Environment
    parser.add_argument("--env", type=str, choices=["dev", "prod", "test"],
                       default="dev", help="Environment to run in")
    
    # Root directory
    parser.add_argument("--root-dir", type=Path,
                       help="Root directory for the project")
    
    # Workflow steps
    parser.add_argument("--prepare-data", action="store_true",
                       help="Prepare data for training")
    parser.add_argument("--train", action="store_true",
                       help="Train the model")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate the model")
    parser.add_argument("--export", action="store_true",
                       help="Export the model")
    
    return parser.parse_args()

def main():
    """Main function to run the workflow."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Create workflow manager
        workflow = WorkflowManager.from_config_files(
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            data_config_path=args.data_config,
            task_specific_config_path=args.task_config,
            environment=args.env,
            root_dir=args.root_dir
        )
        
        # Run workflow steps
        if args.prepare_data:
            workflow.prepare_data()
        
        if args.train:
            workflow.train()
        
        if args.evaluate:
            metrics = workflow.evaluate()
            logger.info(f"Evaluation metrics: {metrics}")
        
        if args.export:
            workflow.export_model()
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise

if __name__ == "__main__":
    main() 