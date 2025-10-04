import Monsoon.HVPM as HVPM
import Monsoon.LVPM as LVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op
import Monsoon.pmapi as pmapi
import numpy as np
import time

def testHVPM(serialno=None,Protocol=pmapi.USB_protocol()):
    HVMON = HVPM.Monsoon()
    HVMON.setup_usb(serialno,Protocol)
    print("HVPM Serial Number: " + repr(HVMON.getSerialNumber()))
    HVMON.setPowerUpCurrentLimit(8)
    HVMON.setRunTimeCurrentLimit(8)
    HVMON.fillStatusPacket()
    #HVMON.calibrateVoltage()
    HVMON.setVout(4.2)
    HVengine = sampleEngine.SampleEngine(HVMON)
    #Output to CSV
    HVengine.enableCSVOutput("HV Main Example.csv")
    #Turning off periodic console outputs.
    HVengine.ConsoleOutput(True)

    #Setting all channels enabled
    HVengine.enableChannel(sampleEngine.channels.MainCurrent)
    HVengine.enableChannel(sampleEngine.channels.MainVoltage)

    #Setting trigger conditions
    numSamples=sampleEngine.triggers.SAMPLECOUNT_INFINITE

    HVengine.setStartTrigger(sampleEngine.triggers.GREATER_THAN,0) 
    HVengine.setStopTrigger(sampleEngine.triggers.GREATER_THAN,numSamples)
    HVengine.setTriggerChannel(sampleEngine.channels.timeStamp) 

    #Actually start collecting samples
    HVengine.startSampling(numSamples,1)


def main():
    HVPMSerialNo = 26589  # 替换为实际的序列号
    while(True):
        protocol = pmapi.USB_protocol()
        testHVPM(HVPMSerialNo, protocol)
        time.sleep(60)
    # try:
    #     protocol = pmapi.USB_protocol()
    #     protocol.Connect('HVPM', HVPMSerialNo)
    #     if protocol.DEVICE is None:
    #         print("Unable to find device")
    #         return
    #     testHVPM(HVPMSerialNo, protocol)
    # except Exception as e:
    #     print(f"Error in main: {e}")

if __name__ == "__main__":
    main()