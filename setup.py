from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read();

def license():
    with open('LICENSE') as f:
        return f.read();

setup(
    name='abalone-ring-classification',
    version='0.0.1',
    description='K Nearest Neighbour classification of Abalone Rings',
    long_description=readme(),
    url='https://github.com/eyeonechi/abalone-ring-classification',
    author='Ivan Ken Weng Chee',
    author_email='ichee@student.unimelb.edu.au',
    license=license(),
    keywords=[
        'COMP10001'
    ],
    scripts=[
        'src/abalone_ring_classifier.py'
    ],
    packages=[],
    zip_safe=False,
    include_package_data=True
)
