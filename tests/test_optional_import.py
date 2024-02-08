from flexdat.utils_import import optional_import


def test_optional_import_success():
    m = optional_import('flexdat.utils_import')
    assert m.optional_import is not None


def test_optional_import_failed():
    # import a module that doesn't exist. Error should not be raised
    # until the module is being used (but not imported)
    m = optional_import('module.doesnot.exist')
    assert m is not None

    exception_raised = False
    try:
        m.function_call()
    except Exception:
        exception_raised = True
    assert exception_raised
