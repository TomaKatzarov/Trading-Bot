@echo off
REM Quick test runner for Windows
REM Usage: tests\quick_test.bat

echo ==============================================
echo Running Comprehensive Training Pipeline Tests
echo ==============================================
echo.

REM Activate virtual environment if exists
if exist "activate_rl_env.bat" (
    echo Activating virtual environment...
    call activate_rl_env.bat
)

REM Run tests
python tests/run_comprehensive_tests.py %*

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ All tests passed! Training pipeline is ready.
) else (
    echo.
    echo ❌ Some tests failed. Review output above.
)

exit /b %ERRORLEVEL%
