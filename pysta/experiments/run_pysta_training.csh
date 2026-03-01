#!/bin/csh
#
# run_pysta_training.csh
# Run CircuitNet GNN training using PySTA data loader
#
# Usage:
#   ./run_pysta_training.csh [data_path] [checkpoint_name] [iterations]
#
# Example:
#   ./run_pysta_training.csh ./data/zipcpu pysta_cp1 5000
#

# Default values
set DATA_PATH = "./data/zipcpu"
set CHECKPOINT = "pysta_checkpoint"
set ITERATIONS = 5000
set TRAIN_NUM = 1
set TEST_NUM = 0
set BATCH_SIZE = 1
set LR = 0.001

# Parse command line arguments
if ($#argv >= 1) then
    set DATA_PATH = $1
endif
if ($#argv >= 2) then
    set CHECKPOINT = $2
endif
if ($#argv >= 3) then
    set ITERATIONS = $3
endif

# Set paths (relative to project root)
set PYSTA_ROOT = `dirname $0`/..
set EXPERIMENT_DIR = "${PYSTA_ROOT}/pysta/experiments"
set PYTHON = "python3"

# Print configuration
echo "=============================================="
echo "PySTA CircuitNet Training"
echo "=============================================="
echo "Data Path:    $DATA_PATH"
echo "Checkpoint:   $CHECKPOINT"
echo "Iterations:   $ITERATIONS"
echo "Train/Test:   $TRAIN_NUM / $TEST_NUM"
echo "Batch Size:   $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "=============================================="

# Change to experiment directory
cd $EXPERIMENT_DIR

# Run training
echo ""
echo "Starting training..."
echo ""

$PYTHON train.py \
    --data_path $DATA_PATH \
    --checkpoint $CHECKPOINT \
    --iteration $ITERATIONS \
    --train_data_number $TRAIN_NUM \
    --test_data_number $TEST_NUM \
    --batch_size $BATCH_SIZE \
    --lr $LR

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoint saved to: ${EXPERIMENT_DIR}/checkpoints/${CHECKPOINT}/"
echo "=============================================="
