# test_e2e.py

import pytest
import subprocess
import sys
from playwright.sync_api import sync_playwright, expect
import os
import time

@pytest.fixture(scope="session", autouse=True)
def run_streamlit_app():
    """Launch Streamlit app before tests and kill it after."""
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", 
         "pyAudioAnalysis/audioUI.py",
         "--server.port=8504",
         "--server.headless=true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the app a few seconds to start up
    time.sleep(10)
    # Yield to run the tests
    yield
    # Terminate the streamlit process
    process.terminate()
    process.wait()

def test_file_upload():
    """Test file upload functionality."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8504")
        
        # Verify file uploader exists
        file_uploader = page.locator('[data-testid="stFileUploaderDropzoneInput"]')
        expect(file_uploader).to_be_attached(timeout=10000)
        
        # Upload test file
        test_file = os.path.join(os.path.dirname(__file__), "../pyAudioAnalysis/data/doremi.wav")
        file_uploader.set_input_files(test_file)
        
        # Wait for file to be processed
        page.wait_for_timeout(1000)
        
        # Verify file was uploaded
        expect(page.get_by_text("doremi.wav")).to_be_visible()
        
        # Verify analysis type selector using more specific selector
        analysis_selector = page.locator('[data-testid="stSelectbox"] >> text=Select Analysis Type')
        expect(analysis_selector).to_be_visible()
        
        # Verify we can select an option
        analysis_input = page.locator('[role="combobox"][aria-label*="Select Analysis Type"]')
        expect(analysis_input).to_be_visible()
        
        browser.close()

def test_debug_mode_visibility():
    """Test that debug mode checkbox is visible and interactive in sidebar."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8504")
        
        # Check debug section header
        debug_header = page.locator('[data-testid="stMarkdownContainer"] h2:text("Debug Settings")')
        expect(debug_header).to_be_visible()
        
        # Check debug checkbox container and label
        debug_container = page.locator('[data-testid="stCheckbox"]')
        expect(debug_container).to_be_visible()
        
        # Check debug label
        debug_label = page.locator('[data-testid="stMarkdownContainer"] p:text("Enable Debug Mode")')
        expect(debug_label).to_be_visible()
        
        # Click the checkbox container to toggle
        debug_container.click()
        
        # Verify checkbox state changed
        expect(page.locator('input[aria-checked="true"]')).to_be_attached()
        
        browser.close()

