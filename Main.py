import multiprocessing
from multiprocessing import Queue
import os
from Wifi import wifi
from Wifi import CommandLine


# WIFI Receive
def wifiReceive(rq, ipq):
    while True:
        data, ip = wf.wifiReceive()
        rq.put(data.decode("utf-8"))
        ipq.put(ip)

# WIFI Send
def wifiSend(sq, ipq):
    iphoneIP = None
    while True:
        if not ipq.empty():
            iphoneIP = ipq.get()

        if not sq.empty():
            data = sq.get()
            wf.wifiSend(data, iphoneIP)

# parse the command from WIFI
def parseCommand(rq, sq):
    # Create a Command instance
    cm = CommandLine.command()

    while True:
        if not rq.empty():
            getCommand = rq.get()
            message = cm.parseCommand(getCommand)
            if message:
                #Send data to iPhone
                sq.put(message)

#Read NVIDIA Jetson TX2 GPU status
def readGPU(sq):
    cmd = "echo \"nvidia\" | sudo -S -k  /home/nvidia/tegrastats"
    p = os.popen(cmd)
    print(p.readline())
    while True:
        s = ('GPUSTATUS'+p.readline()).encode()
        sq.put(s)

# Open the tensor board service for split model
def loadSplitTensorBoard():
    cmd = "python3 /home/nvidia/.local/lib/python3.5/site-packages/tensorboard/main.py" \
          " --logdir=SP:/home/nvidia/DeepSensing/SPModel/logs,MP:/home/nvidia/DeepSensing/MPModel/logs"
    os.system(cmd)

# Open the HTTPServer for Share file
def openHTTPServer():
    cmd = "cd /home/nvidia/DeepSensing/Share && python3 -m http.server 8082"
    os.system(cmd)

if __name__ == '__main__':

    #Create a WIFI received queue
    #Receive data from iPhone
    rq = Queue()
    #Create a WIFI send queue
    #Send data to iPhone
    sq = Queue()
    #Create a iPhone ip queue
    ipq = Queue()

    # Create a WifiSocket instance
    wf = wifi.WifiSocket()
    # Initialize the WifiSocket
    wf.__int__()


    # Create concurrent task
    wifi_receive = multiprocessing.Process(name='wifiReceive', target=wifiReceive, args=(rq,ipq))
    wifi_send = multiprocessing.Process(name='wifiSend', target=wifiSend, args=(sq,ipq))
    parse_command = multiprocessing.Process(name='parseCommand', target=parseCommand, args=(rq, sq))
    # load_split_tensor_board = multiprocessing.Process(name='loadSplitTensorBoard', target=loadSplitTensorBoard)
    read_gpu = multiprocessing.Process(name='readGPU', target=readGPU, args=(sq,))
    open_HTTPServer = multiprocessing.Process(name='openHTTPServer', target=openHTTPServer)

    wifi_receive.start()
    wifi_send.start()
    read_gpu.start()
    parse_command.start()
    # load_split_tensor_board.start()
    open_HTTPServer.start()

    wifi_receive.join()
    wifi_send.join()
    read_gpu.join()
    parse_command.join()
    # load_split_tensor_board.join()
    open_HTTPServer.join()




