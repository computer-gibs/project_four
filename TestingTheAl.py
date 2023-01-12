import warnings
def api_v1():
    warnings.warn(UserWarning("Image size must not be over 1000x1000 pixels"))
    return 1
def test_one():
    assert api_v1() == 1
def api_v2():
    warnings.warn(UserWarning("Wrong file format"))
    return 1
def test_two():
    assert api_v2() == 1
def api_v3():
    warnings.warn(UserWarning("Image with alpha channel"))
    return 1
def test_three():
    assert api_v3() == 1