#!/bin/bash
# Test runner script for SAM3 Roto Ultimate

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "SAM3 Roto Ultimate - Test Runner"
echo -e "========================================${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest not found${NC}"
    echo "Installing pytest..."
    pip install pytest pytest-cov pytest-xdist
fi

# Parse arguments
RUN_MODE="all"
VERBOSE=""
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_MODE="unit"
            shift
            ;;
        --integration)
            RUN_MODE="integration"
            shift
            ;;
        --fast)
            RUN_MODE="fast"
            shift
            ;;
        --slow)
            RUN_MODE="slow"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        --cov|--coverage)
            COVERAGE="--cov=sam3roto --cov-report=html --cov-report=term"
            shift
            ;;
        --parallel)
            PARALLEL="-n auto"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run tests based on mode
case $RUN_MODE in
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        pytest tests/ -m "unit" $VERBOSE $COVERAGE $PARALLEL
        ;;
    integration)
        echo -e "${BLUE}Running integration tests...${NC}"
        pytest tests/ -m "integration" $VERBOSE $COVERAGE
        ;;
    fast)
        echo -e "${BLUE}Running fast tests (excluding slow)...${NC}"
        pytest tests/ -m "not slow" $VERBOSE $COVERAGE $PARALLEL
        ;;
    slow)
        echo -e "${BLUE}Running slow tests...${NC}"
        pytest tests/ -m "slow" $VERBOSE $COVERAGE
        ;;
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        pytest tests/ $VERBOSE $COVERAGE
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

# Show coverage report location if generated
if [ -n "$COVERAGE" ]; then
    echo -e "${BLUE}Coverage report generated:${NC}"
    echo "  HTML: htmlcov/index.html"
    echo "  Open with: firefox htmlcov/index.html"
fi
