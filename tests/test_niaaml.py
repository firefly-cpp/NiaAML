from niaaml import __version__
from niaaml.data import CSVDataReader

def test_version():
    assert __version__ == '0.1.0'

test = CSVDataReader(src='E:/School workspace/Magistrska naloga 2/abalone.csv')
test.readData()