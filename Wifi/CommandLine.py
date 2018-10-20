import os
from SingleParticle import  TestSPModel as tsp
from MultiParticle import TestMPModel as tmp

loadSPModel = 0
loadMPModel = 0

class command:
    def parseCommand(self, wifiReceiveCommand):

        global loadSPModel
        global sessSP
        global trainNetSP

        global loadMPModel
        global sessMP
        global trainNetMP

        # Shut down Jetson TX2
        if "SHUTDOWN" in wifiReceiveCommand:
            cmd = "echo \"nvidia\" | sudo -S -k poweroff"
            os.system(cmd)

            return False

        # Restart Jetson TX2
        if "RESTART" in wifiReceiveCommand:
            cmd = "echo \"nvidia\" | sudo -S -k reboot"
            os.system(cmd)

            return False

        # Test single particle model for one frame
        if "TEST1" in wifiReceiveCommand:

            # Only load single particle model for the first time
            if loadSPModel==0 and loadMPModel==0:
                sessSP, trainNetSP = tsp.loadSPModel()
                loadSPModel = 1
                message = tsp.testSingleSPModel(sessSP, trainNetSP)

            elif loadSPModel==0 and loadMPModel==1:
                tmp.closeMPModel()
                loadSPModel = 1
                loadMPModel = 0
                sessSP, trainNetSP = tsp.loadSPModel()
                message = tsp.testSingleSPModel(sessSP, trainNetSP)

            else:
                message = tsp.testSingleSPModel(sessSP, trainNetSP)

            return message

        # Test single particle model for one frame
        if "TEST2" in wifiReceiveCommand:

            # Only load single particle model for the first time

            if loadSPModel == 0 and loadMPModel == 0:
                sessSP, trainNetSP = tsp.loadSPModel()
                loadSPModel = 1
                message = tsp.testBatchSPModel(sess=sessSP, trainNet=trainNetSP, batch_size=200)

            elif loadSPModel == 0 and loadMPModel == 1:
                tmp.closeMPModel()
                loadSPModel = 1
                loadMPModel = 0
                sessSP, trainNetSP = tsp.loadSPModel()
                message = tsp.testBatchSPModel(sess=sessSP, trainNet=trainNetSP, batch_size=200)

            else:
                message = tsp.testBatchSPModel(sess=sessSP, trainNet=trainNetSP, batch_size=200)

            return message

        # Test multi particle model for one frame
        if "TEST3" in wifiReceiveCommand:

            # Only load multi particle model for the first time
            if loadMPModel == 0 and loadSPModel == 0:
                sessMP, trainNetMP = tmp.loadMPModel()
                loadMPModel = 1
                message = tmp.testSingleMPModel(sessMP, trainNetMP)

            elif loadMPModel == 0 and loadSPModel == 1:
                tsp.closeSPModel()
                loadMPModel = 1
                loadSPModel = 0
                sessMP, trainNetMP = tmp.loadMPModel()
                message = tmp.testSingleMPModel(sessMP, trainNetMP)

            else:
                message = tmp.testSingleMPModel(sessMP, trainNetMP)

            return message

        # Test multi particle model for one frame
        if "TEST4" in wifiReceiveCommand:

            # Only load multi particle model for the first time
            if loadMPModel == 0 and loadSPModel == 0:
                sessMP, trainNetMP = tmp.loadMPModel()
                loadMPModel = 1
                message = tmp.testBatchMPModel(sess=sessMP, trainNet=trainNetMP, batch_size=200)

            elif loadMPModel == 0 and loadSPModel == 1:
                tsp.closeSPModel()
                loadMPModel = 1
                loadSPModel = 0
                sessMP, trainNetMP = tmp.loadMPModel()
                message = tmp.testBatchMPModel(sess=sessMP, trainNet=trainNetMP, batch_size=200)

            else:
                message = tmp.testBatchMPModel(sess=sessMP, trainNet=trainNetMP, batch_size=200)

            return message

        # Set LAN Interface IP
        if "SETLAN" in wifiReceiveCommand:
            cmd = "echo \"nvidia\" | sudo -S -k ifconfig eth0 192.168.10.3 netmask 255.255.255.0"
            os.system(cmd)

            return False










