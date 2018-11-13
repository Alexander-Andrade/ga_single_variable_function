install:
pip install -r requirements.txt

tests:

python -m unittests 
python -m unittest unittests.GaTests
python -m unittest unittests.GaTests.test_chromosome_length