#!/bin/bash
#####################################################################
# Experiment 6 - Pod 3: Adaptive Layer 4
# 
# This pod tests the baseline configuration with all layers enabled
# but NO inter-layer coordination or adaptive behavior.
#####################################################################

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         EXPERIMENT 6 - POD 3: ADAPTIVE LAYER 4                ║"
echo "║         Enhanced Monitoring Triggered by Upstream Risk                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CONFIG="adaptive_l4"
TRIALS=5
EXPECTED_TRACES=210  # 42 attacks × 5 trials

echo "📋 Configuration: $CONFIG"
echo "🔢 Trials per attack: $TRIALS"
echo "📊 Expected traces: $EXPECTED_TRACES"
echo ""

# =================================================================
# Step 1: System Check
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: System Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found"
    exit 1
fi
echo "✅ pip3: $(pip3 --version)"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found - installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi
echo "✅ Ollama: $(ollama --version)"

echo ""

# =================================================================
# Step 2: Install Dependencies
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Installing Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip3 install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || true
echo "✅ Python packages installed"
echo ""

# =================================================================
# Step 3: Setup Ollama + llama3
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Setting up Ollama with llama3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start Ollama service
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!
echo "✅ Ollama service started (PID: $OLLAMA_PID)"
sleep 3

# Pull llama3 model
echo "📥 Pulling llama3 model (this may take a few minutes)..."
ollama pull llama3
echo "✅ llama3 model ready"
echo ""

# =================================================================
# Step 4: Run Experiment
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Running Experiment 6 - ADAPTIVE L4 Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  Start time: $(date)"
echo ""

# Create results directory
mkdir -p results

# Determine if running in background or foreground
BACKGROUND=${BACKGROUND:-false}

if [ "$BACKGROUND" = "true" ]; then
    echo "🔄 Starting experiment in BACKGROUND mode..."
    echo "   Log file: results/experiment.log"
    echo "   PID file: results/experiment.pid"
    echo ""
    
    # Run in background with nohup
    nohup python3 run_experiment6_coordination.py \
        --config $CONFIG \
        --output results \
        --trials $TRIALS > results/experiment.log 2>&1 &
    
    EXPERIMENT_PID=$!
    echo $EXPERIMENT_PID > results/experiment.pid
    
    echo "✅ Experiment started in background!"
    echo "   Process ID: $EXPERIMENT_PID"
    echo ""
    echo "📋 To monitor progress:"
    echo "   tail -f results/experiment.log"
    echo ""
    echo "📊 To check status:"
    echo "   ps aux | grep $EXPERIMENT_PID"
    echo ""
    echo "📈 To watch trace count:"
    echo "   watch -n 10 'sqlite3 results/exp6_$CONFIG.db \"SELECT COUNT(*) FROM execution_traces\"'"
    echo ""
    echo "⏹️  To stop (if needed):"
    echo "   kill $EXPERIMENT_PID"
    echo ""
    echo "Expected completion: ~3-5 minutes"
    echo "You can safely disconnect - the experiment will continue running."
    echo ""
    exit 0
    
else
    echo "▶️  Running experiment in FOREGROUND mode..."
    echo "   (Set BACKGROUND=true to run in background)"
    echo ""
    
    python3 run_experiment6_coordination.py \
        --config $CONFIG \
        --output results \
        --trials $TRIALS 2>&1 | tee results/experiment.log

    EXIT_CODE=$?
    echo ""
    echo "⏱️  End time: $(date)"
    echo ""

    # Check results
    if [ $EXIT_CODE -eq 0 ]; then
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║                  ✅ EXPERIMENT COMPLETE - POD 3                ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "📦 Results Location:"
        echo "  • Database: results/exp6_$CONFIG.db"
        echo "  • Summary:  results/exp6_${CONFIG}_summary.json"
        echo "  • Log:      results/experiment.log"
        echo ""
        
        # Display summary if available
        if [ -f "results/exp6_${CONFIG}_summary.json" ]; then
            echo "📊 Experiment Summary:"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            cat "results/exp6_${CONFIG}_summary.json"
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        fi
        
        echo ""
        echo "📥 NEXT STEPS:"
        echo "  1. Download results files from this pod"
        echo "  2. Repeat on remaining pods (adaptive_l3, adaptive_l4, full_adaptive)"
        echo "  3. Run cross-pod analysis to compare configurations"
        echo ""
        
        exit 0
    else
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║                  ❌ EXPERIMENT FAILED - POD 3                  ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Exit code: ${EXIT_CODE}"
        echo "Check results/experiment.log for error details"
        exit ${EXIT_CODE}
    fi
fi
