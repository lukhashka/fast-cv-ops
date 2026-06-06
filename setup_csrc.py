from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fast_cv_ops',
    ext_modules=[
        CppExtension(
            name='fast_cv_ops', # Так наш модуль буде називатися в Python
            sources=['csrc/iou.cpp'], # Вказуємо шлях до нашого C++ файлу
            extra_compile_args=['-O3'] # Вмикаємо ту саму максимальну оптимізацію
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)