def test_app_main_is_callable():

    import app  # noqa: F401

    assert hasattr(app, "main")
    assert callable(app.main)
