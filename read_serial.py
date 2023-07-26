import serial
from collections import deque
ser = serial.Serial("COM6", 115200)
instring = ""
imu_data = [0,0,0]
# data_x = deque(maxlen=10000)
# data_y = deque(maxlen=10000)
# data_z = deque(maxlen=10000)
# cur_chan = 0
while True:
    # try:
    in_char = ser.read()
    # print(instring)
    # print(in_char)
    if in_char != b'\n' and in_char != b'\r':
        instring += chr(ord(in_char))
    elif in_char == b"\n":
        if len(instring) != 0:
            # print(instring)
            chans = instring.split(',')
            for i, chan in enumerate(chans):
                imu_data[i] = float(chan)
            instring = ""
            # x,y,z = imu_data
            # data_x.append(x)
            # data_y.append(y)
            # data_z.append(z)
    print(imu_data)
        # if in_char == '\n'
    # except KeyboardInterrupt:
    #     break
print("program execution terminated")