#!/bin/bash
# Docker Build and Test Script for Adobe India Hackathon
# =====================================================

echo "🐳 ADOBE INDIA HACKATHON - DOCKER BUILD & TEST"
echo "=============================================="

# Configuration
IMAGE_NAME="adobe-heading-detector"
CONTAINER_NAME="adobe-test-container"
INPUT_DIR="./input"
OUTPUT_DIR="./output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker."
        exit 1
    fi
    
    log_success "Docker is available and running"
}

# Create test directories
setup_directories() {
    log_info "Setting up test directories..."
    
    mkdir -p "$INPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # Clean output directory
    rm -rf "$OUTPUT_DIR"/*
    
    log_success "Directories created: $INPUT_DIR, $OUTPUT_DIR"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    if docker build --platform linux/amd64 -t "$IMAGE_NAME" .; then
        log_success "Docker image built successfully: $IMAGE_NAME"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Test Docker image
test_image() {
    log_info "Testing Docker image..."
    
    # Check if there are any PDF files in input directory
    if [ ! "$(ls -A $INPUT_DIR/*.pdf 2>/dev/null)" ]; then
        log_warning "No PDF files found in $INPUT_DIR"
        log_info "Creating test directory structure..."
        return 0
    fi
    
    # Run the container
    log_info "Running container with competition parameters..."
    
    if docker run --rm \
        -v "$(pwd)/$INPUT_DIR":/app/input:ro \
        -v "$(pwd)/$OUTPUT_DIR":/app/output \
        --network none \
        "$IMAGE_NAME"; then
        log_success "Container executed successfully"
    else
        log_error "Container execution failed"
        return 1
    fi
    
    # Check output
    if [ "$(ls -A $OUTPUT_DIR/*.json 2>/dev/null)" ]; then
        log_success "JSON output files generated:"
        ls -la "$OUTPUT_DIR"/*.json
    else
        log_warning "No JSON output files found"
    fi
}

# Inspect image
inspect_image() {
    log_info "Inspecting Docker image..."
    
    echo ""
    log_info "Image size:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    echo ""
    log_info "Image layers:"
    docker history "$IMAGE_NAME" --format "table {{.CreatedBy}}\t{{.Size}}"
}

# Run interactive container for debugging
debug_container() {
    log_info "Starting interactive container for debugging..."
    
    docker run -it --rm \
        -v "$(pwd)/$INPUT_DIR":/app/input:ro \
        -v "$(pwd)/$OUTPUT_DIR":/app/output \
        --entrypoint /bin/bash \
        "$IMAGE_NAME"
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove dangling images
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     - Build the Docker image"
    echo "  test      - Test the Docker image"
    echo "  inspect   - Inspect the Docker image"
    echo "  debug     - Start interactive container"
    echo "  cleanup   - Clean up Docker resources"
    echo "  all       - Run build and test (default)"
    echo ""
    echo "Example:"
    echo "  $0 build"
    echo "  $0 test"
    echo "  $0 all"
}

# Main execution
main() {
    local command="${1:-all}"
    
    case "$command" in
        "build")
            check_docker
            setup_directories
            build_image
            ;;
        "test")
            check_docker
            setup_directories
            test_image
            ;;
        "inspect")
            check_docker
            inspect_image
            ;;
        "debug")
            check_docker
            setup_directories
            debug_container
            ;;
        "cleanup")
            check_docker
            cleanup
            ;;
        "all")
            check_docker
            setup_directories
            build_image
            test_image
            inspect_image
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"

echo ""
log_info "Docker script completed."
echo ""
echo "📋 QUICK REFERENCE:"
echo "Build:  docker build --platform linux/amd64 -t adobe-heading-detector ."
echo "Run:    docker run --rm -v \$(pwd)/input:/app/input:ro -v \$(pwd)/output:/app/output --network none adobe-heading-detector"
echo "Debug:  docker run -it --rm -v \$(pwd)/input:/app/input:ro -v \$(pwd)/output:/app/output --entrypoint /bin/bash adobe-heading-detector"
