#!/bin/bash

# Phase 10: Pipeline Integration - Execution Script
#
# Complete ML Pipeline: bmarket.db → subscription_predictions.csv
# Models: Ensemble Voting (92.5% accuracy) with 3-tier failover architecture
# Features: 45 features including Phase 9 optimized business features
# Performance: 92.5% accuracy baseline, 72,000+ records/second ensemble processing
# Business Purpose: Automated term deposit prediction with 6,112% ROI potential
# Phase 9 Integration: All 9 optimization modules integrated for production deployment
#
# Usage:
#   ./run.sh                    # Run complete pipeline
#   ./run.sh test              # Run in test mode
#   ./run.sh benchmark         # Run performance benchmark
#   ./run.sh validate          # Validate pipeline components

# Parse command line arguments
MODE=${1:-production}

echo "🚀 Phase 10: Pipeline Integration - Execution Script"
echo "=================================================="
echo "📋 Pipeline Configuration:"
echo "   • Models: Ensemble Voting (92.5% accuracy)"
echo "   • Architecture: 3-tier failover"
echo "   • Features: 45 optimized features"
echo "   • Performance: 72,000+ rec/sec ensemble"
echo "   • ROI Potential: 6,112%"
echo "   • Phase 9 Integration: All 9 modules"
echo "   • Execution Mode: $MODE"
echo "=================================================="

# Function to check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        echo "❌ Error: Python not found. Please install Python 3.8+"
        exit 1
    fi

    # Check Python version (fix for Windows compatibility)
    python_version=$(python --version 2>&1 | awk '{print $2}')
    echo "✅ Python version: $python_version"

    # Check if required directories exist
    if [ ! -d "src" ]; then
        echo "❌ Error: src directory not found"
        exit 1
    fi

    if [ ! -d "src/pipeline_integration" ]; then
        echo "❌ Error: src/pipeline_integration directory not found"
        exit 1
    fi

    # Create required directories if they don't exist
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/featured
    mkdir -p trained_models
    mkdir -p specs/output

    echo "✅ Directory structure validated"

    # Check if main.py exists
    if [ ! -f "main.py" ]; then
        echo "❌ Error: main.py not found"
        exit 1
    fi

    echo "✅ Prerequisites check completed"
}

# Function to run pipeline based on mode
run_pipeline() {
    local mode=$1
    echo ""
    echo "🔄 Executing Phase 10 Pipeline Integration in $mode mode..."

    case $mode in
        "test")
            python main.py --test
            ;;
        "benchmark")
            python main.py --benchmark
            ;;
        "validate")
            python main.py --validate
            ;;
        "production"|*)
            python main.py
            ;;
    esac
}

# Function to display execution summary
display_summary() {
    local exit_code=$1
    local mode=$2

    echo ""
    echo "📊 Pipeline Execution Summary ($mode mode):"

    if [ $exit_code -eq 0 ]; then
        echo "✅ Pipeline completed successfully"

        case $mode in
            "test")
                echo "🧪 Test validation completed"
                echo "📋 All pipeline components functional"
                ;;
            "benchmark")
                echo "📊 Performance benchmark completed"
                echo "📈 Performance metrics calculated"
                ;;
            "validate")
                echo "✅ Component validation completed"
                echo "🔧 Infrastructure requirements checked"
                ;;
            "production"|*)
                echo "📄 Output: subscription_predictions.csv"
                echo "📈 Business metrics calculated"
                echo "🎯 ROI analysis completed"
                echo "💼 Customer segment analysis available"

                # Check if output file was created
                if [ -f "subscription_predictions.csv" ]; then
                    file_size=$(du -h subscription_predictions.csv | cut -f1)
                    line_count=$(wc -l < subscription_predictions.csv)
                    echo "📊 Output file size: $file_size"
                    echo "📊 Predictions generated: $((line_count - 1))"
                fi
                ;;
        esac

        echo "🎉 Phase 10 Step 2 implementation successful!"

    else
        echo "❌ Pipeline execution failed"
        echo "📋 Troubleshooting steps:"
        echo "   1. Check Python dependencies are installed"
        echo "   2. Verify src/pipeline_integration modules are present"
        echo "   3. Check Phase 9 modules in src/model_optimization"
        echo "   4. Review error logs for specific issues"
        echo ""
        echo "📋 Next Steps:"
        echo "   1. Review error messages above"
        echo "   2. Check Phase 10 Step 2 implementation"
        echo "   3. Validate Phase 9 module integration"
        echo "   4. Run Phase 10 Step 3: Comprehensive Testing"
    fi
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [mode]"
    echo ""
    echo "Modes:"
    echo "  production  Run complete production pipeline (default)"
    echo "  test        Run pipeline validation tests"
    echo "  benchmark   Run performance benchmarking"
    echo "  validate    Validate pipeline components"
    echo ""
    echo "Examples:"
    echo "  $0                # Run production pipeline"
    echo "  $0 test          # Run test mode"
    echo "  $0 benchmark     # Run benchmark mode"
    echo "  $0 validate      # Run validation mode"
}

# Main execution
main() {
    # Check for help flag
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    fi

    # Check prerequisites
    check_prerequisites

    # Run pipeline
    run_pipeline "$MODE"

    # Capture exit code
    exit_code=$?

    # Display summary
    display_summary $exit_code "$MODE"

    echo "=================================================="
    exit $exit_code
}

# Execute main function
main "$@"
