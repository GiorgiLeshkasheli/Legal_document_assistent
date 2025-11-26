from src.utils import get_logger, load_environment


def test_get_logger_returns_logger():
    logger = get_logger("test_logger")
    # Basic checks
    assert logger.name == "test_logger"
    # Should not crash when logging
    logger.info("This is a test log from test_get_logger.")


def test_load_environment_does_not_crash():
    # Just ensure it runs without error
    load_environment()
