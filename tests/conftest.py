def pytest_configure(config):
    """Configure pytest."""
    # Register custom marks
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )