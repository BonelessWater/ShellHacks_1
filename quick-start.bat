@echo off
REM quick-start.bat - Start your ShellHacks Invoice Frontend with Docker (Windows)

echo 🚀 Starting ShellHacks Invoice Frontend
echo ==============================================

REM Check if Docker is running
echo 🔍 Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo ✅ Docker is running

REM Check if we're in the right directory
if not exist "docker-compose.yml" (
    echo ❌ docker-compose.yml not found. Please run this script from the project root.
    pause
    exit /b 1
)

if not exist "frontend" (
    echo ❌ frontend directory not found. Please make sure you're in the project root.
    pause
    exit /b 1
)

REM Create .env if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env file...
    (
        echo # Frontend Environment Variables
        echo REACT_APP_API_URL=http://localhost:8000
        echo REACT_APP_WEBSOCKET_URL=ws://localhost:8000
        echo REACT_APP_ENVIRONMENT=development
        echo.
        echo # Optional: Backend API Key ^(uncomment if needed^)
        echo # GOOGLE_API_KEY_0=your_google_api_key_here
    ) > .env
    echo ✅ .env file created
) else (
    echo ✅ .env file exists
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down >nul 2>&1

REM Build and start the frontend
echo 🔨 Building and starting frontend...
echo This may take a few minutes on first run...

REM Build with output
docker-compose build frontend

REM Start the service
echo 🚀 Starting the frontend service...
docker-compose up -d frontend

REM Wait for the service to be ready
echo ⏳ Waiting for frontend to be ready...
timeout /t 10 /nobreak >nul

REM Check if service is running
docker-compose ps frontend | find "Up" >nul
if %errorlevel% equ 0 (
    echo.
    echo ==============================================
    echo 🎉 SUCCESS! Frontend is running
    echo.
    echo 📱 Frontend URL: http://localhost:3000
    echo 🐳 Docker Status: Running
    echo.
    echo 📋 Useful Commands:
    echo   View logs:    docker-compose logs -f frontend
    echo   Stop:         docker-compose down
    echo   Restart:      docker-compose restart frontend
    echo.
    echo ✨ Your ShellHacks Invoice app is ready!
    echo ==============================================
    echo.
    
    REM Ask if user wants to see logs
    set /p choice="Would you like to view the logs? (y/n): "
    if /i "%choice%"=="y" (
        echo 📊 Showing logs ^(Press Ctrl+C to exit^):
        docker-compose logs -f frontend
    )
) else (
    echo ❌ Frontend failed to start. Check the logs:
    docker-compose logs frontend
    pause
    exit /b 1
)

pause