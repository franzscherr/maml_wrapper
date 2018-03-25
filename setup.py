from setuptools import setup, find_packages


setup(name='maml_wrapper',
      version='0.1dev0',
      description='Wrapper around models to enable model-agnostic meta learning',
      keywords='meta-learning learning to learn tensorflow machine learning',
      url='https://github.com/franzscherr/maml_wrapper',
      author='Franz Scherr',
      license='MIT',
      packages=find_packages(),
      install_requires=open('requirements.txt').read().split(),
      include_package_data=True,
      zip_safe=False)
