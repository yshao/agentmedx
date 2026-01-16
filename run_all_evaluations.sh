#!/bin/bash
# Run all 6 MedAgentBench evaluations with 10 second pauses

echo "=========================================="
echo "Running All MedAgentBench Evaluations"
echo "=========================================="
echo ""

# List of all tasks
tasks=("diabetes_001:diabetes" "diabetes_002:diabetes" "cardiology_001:cardiology" "internal_medicine_001:internal_medicine" "diabetes_003:diabetes" "cardiology_002:cardiology")

for task_info in "${tasks[@]}"; do
    IFS=: read task_id category <<< "$task_info"
    
    echo "=========================================="
    echo "Running: $task_id (Category: $category)"
    echo "=========================================="
    
    # Update config for current task
    sed -i "s/task_id = .*/task_id = $task_id/" config/scenario.toml
    sed -i "s/medical_category = .*/medical_category = $category/" config/scenario.toml
    
    # Run evaluation
    PYTHONPATH=.:tutorial_src python run_benchmark.py --config config/scenario.toml --official-format
    
    echo ""
    echo "Completed: $task_id"
    echo ""
    echo "Waiting 10 seconds before next evaluation..."
    sleep 10
    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
