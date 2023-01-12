import LaunchScript
from PIL import Image
import pytest

#Проверка на допустимость размерности входящего изображения больше 1000х1000 пикселей
def test_size():
    with pytest.raises(ValueError):
        LaunchScript.main(input_image=Image.open(url="https://testit.software/storage/app/media/intro-testit-pro__1390.jpg"))

#Проверка на формат входящего файла
def test_format():
    with pytest.raises(ValueError):
        LaunchScript.main(input_image=Image.open(url="https://www.africau.edu/images/default/sample.pdf"))

#Проверка на картинку с прозрачным фоном
def test_alpha():
    LaunchScript.main(input_image=Image.open(url="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"))

