from pyOpenBCI import import OpenBCIPython
def print_raw(smaple):
    print(sample.channels_data)
board = OpenBCICyton(port="com")