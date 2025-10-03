import json
import os
from pathlib import Path

def split_annotations(project_file, output_folder):
    """
    Split annotations from a Label Studio project export file into individual files.
    
    Args:
        project_file: Path to the project JSON file
        output_folder: Path to the output folder for individual annotation files
    """
    # Read the project file
    with open(project_file, 'r') as f:
        data = json.load(f)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each annotation
    for item in data:
        # Extract the task ID
        task_id = item.get('id')
        
        if task_id is not None:
            # Create output filename based on task ID
            output_file = output_path / f"{task_id}.json"
            
            # Write individual annotation file
            with open(output_file, 'w') as f:
                json.dump(item, f, indent=2)
            
            print(f"Saved annotation {task_id} to {output_file}")
    
    print(f"\nTotal annotations processed: {len(data)}")

if __name__ == "__main__":
    # Define paths
    project_file = "project-10-at-2025-10-02-20-30-8e693c6e.json"
    output_folder = "renamed_new_annotation"
    
    # Split annotations
    split_annotations(project_file, output_folder)