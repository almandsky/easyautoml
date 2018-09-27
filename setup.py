from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='easyautoml',
      version='0.1',
      description='Automl with Featuretools generate features and use tpot to select model',
      long_description=readme(),
      classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='automl meachin-learning tpot featuretools feature-engineering model-selection',
      url='https://github.com/almandsky/featuretools-tpot',
      author='Sky Chen',
      author_email='tianxiong_chen@hotmail.com',
      license='MIT',
      packages=['easyautoml'],
      install_requires=[
          'tpot',
          'featuretools',
          'sklearn',
          'numpy',
          'pandas',
          'subprocess'
      ],
      entry_points = {
        'console_scripts': ['easyautoml=easyautoml.command_line:main'],
      },
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)