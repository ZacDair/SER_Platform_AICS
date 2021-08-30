from setuptools import setup, find_packages

setup(
    name='emo_detect',
    version='1.0',
    packages= find_packages(include=['emo_detect', 'emo_detect.*']),
    url='https://github.com/ZacDair/emo_detect',
    license='',
    author='Zac',
    author_email='zacdair@gmail.com',
    description='',
    install_requires=[
            "numpy==1.19.2",
            "keras==2.6.0",
            "librosa==0.8.1",
            "SpeechRecognition==3.8.1",
            "matplotlib==3.3.2",
            "nltk==3.5",
            "pandas==1.1.2",
            "scipy==1.5.2",
            "scikit-learn==0.23.2"
        ]
)
