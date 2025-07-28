@echo off
REM Docker Build and Test Script for Adobe India Hackathon (Windows)
REM ==============================================================

echo 🐳 ADOBE INDIA HACKATHON - DOCKER BUILD ^& TEST (Windows)
echo =======================================================

REM Configuration
set IMAGE_NAME=adobe-heading-detector
set CONTAINER_NAME=adobe-test-container
set INPUT_DIR=.\input
set OUTPUT_DIR=.\output

REM Check command line argument
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=all

goto :main

:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:check_docker
call :log_info "Checking Docker installation..."
docker --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker is not installed or not in PATH"
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker is not running. Please start Docker."
    exit /b 1
)

call :log_success "Docker is available and running"
goto :eof

:setup_directories
call :log_info "Setting up test directories..."

if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Clean output directory
if exist "%OUTPUT_DIR%\*.json" del /q "%OUTPUT_DIR%\*.json"

call :log_success "Directories created: %INPUT_DIR%, %OUTPUT_DIR%"
goto :eof

:build_image
call :log_info "Building Docker image..."

docker build --platform linux/amd64 -t "%IMAGE_NAME%" .
if errorlevel 1 (
    call :log_error "Failed to build Docker image"
    exit /b 1
)

call :log_success "Docker image built successfully: %IMAGE_NAME%"
goto :eof

:test_image
call :log_info "Testing Docker image..."

REM Check if there are any PDF files in input directory
dir "%INPUT_DIR%\*.pdf" >nul 2>&1
if errorlevel 1 (
    call :log_warning "No PDF files found in %INPUT_DIR%"
    call :log_info "Please add PDF files to test the container"
    goto :eof
)

call :log_info "Running container with competition parameters..."

docker run --rm -v "%cd%\%INPUT_DIR%":/app/input:ro -v "%cd%\%OUTPUT_DIR%":/app/output --network none "%IMAGE_NAME%"
if errorlevel 1 (
    call :log_error "Container execution failed"
    goto :eof
)

call :log_success "Container executed successfully"

REM Check output
dir "%OUTPUT_DIR%\*.json" >nul 2>&1
if not errorlevel 1 (
    call :log_success "JSON output files generated:"
    dir "%OUTPUT_DIR%\*.json"
) else (
    call :log_warning "No JSON output files found"
)
goto :eof

:inspect_image
call :log_info "Inspecting Docker image..."

echo.
call :log_info "Image size:"
docker images "%IMAGE_NAME%" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo.
call :log_info "Image layers:"
docker history "%IMAGE_NAME%" --format "table {{.CreatedBy}}\t{{.Size}}"
goto :eof

:debug_container
call :log_info "Starting interactive container for debugging..."

docker run -it --rm -v "%cd%\%INPUT_DIR%":/app/input:ro -v "%cd%\%OUTPUT_DIR%":/app/output --entrypoint /bin/bash "%IMAGE_NAME%"
goto :eof

:cleanup
call :log_info "Cleaning up..."

docker container prune -f
docker image prune -f

call :log_success "Cleanup completed"
goto :eof

:show_usage
echo Usage: %0 [command]
echo.
echo Commands:
echo   build     - Build the Docker image
echo   test      - Test the Docker image
echo   inspect   - Inspect the Docker image
echo   debug     - Start interactive container
echo   cleanup   - Clean up Docker resources
echo   all       - Run build and test (default)
echo.
echo Example:
echo   %0 build
echo   %0 test
echo   %0 all
goto :eof

:main
if "%COMMAND%"=="build" goto :run_build
if "%COMMAND%"=="test" goto :run_test
if "%COMMAND%"=="inspect" goto :run_inspect
if "%COMMAND%"=="debug" goto :run_debug
if "%COMMAND%"=="cleanup" goto :run_cleanup
if "%COMMAND%"=="all" goto :run_all
if "%COMMAND%"=="help" goto :show_usage
if "%COMMAND%"=="-h" goto :show_usage
if "%COMMAND%"=="--help" goto :show_usage

call :log_error "Unknown command: %COMMAND%"
call :show_usage
exit /b 1

:run_build
call :check_docker
if errorlevel 1 exit /b 1
call :setup_directories
call :build_image
goto :end

:run_test
call :check_docker
if errorlevel 1 exit /b 1
call :setup_directories
call :test_image
goto :end

:run_inspect
call :check_docker
if errorlevel 1 exit /b 1
call :inspect_image
goto :end

:run_debug
call :check_docker
if errorlevel 1 exit /b 1
call :setup_directories
call :debug_container
goto :end

:run_cleanup
call :check_docker
if errorlevel 1 exit /b 1
call :cleanup
goto :end

:run_all
call :check_docker
if errorlevel 1 exit /b 1
call :setup_directories
call :build_image
if errorlevel 1 exit /b 1
call :test_image
call :inspect_image
goto :end

:end
echo.
call :log_info "Docker script completed."
echo.
echo 📋 QUICK REFERENCE:
echo Build:  docker build --platform linux/amd64 -t adobe-heading-detector .
echo Run:    docker run --rm -v "%cd%\input":/app/input:ro -v "%cd%\output":/app/output --network none adobe-heading-detector
echo Debug:  docker run -it --rm -v "%cd%\input":/app/input:ro -v "%cd%\output":/app/output --entrypoint /bin/bash adobe-heading-detector
