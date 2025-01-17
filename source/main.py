from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import configparser
import keyboard
import listener
import models
import pandas
import torch
import time
import os

config = configparser.ConfigParser()
config.read('config.ini')
programMode = int(config['General']['programMode'])                   # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
windowSize = int(config['General']['windowSize'])                     # Time between batch predictions and file saves
captureKeyboard = int(config['Collection']['captureKeyboard'])
captureMouse = int(config['Collection']['captureMouse'])
captureController = int(config['Collection']['captureController'])    # DO NOT ENABLE AT SAME TIME AS KB OR MOUSE
downSampleFactor = int(config['Collection']['downSampleFactor'])      # Downsampling factor for mouse events
dataLabel = config['Collection']['dataLabel']                         # control, cheat
killKey = config['Collection']['killKey']
dataType = config['Training']['dataType']
displayGraph = int(config['Analysis']['displayGraph'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController, downSampleFactor, (0))

def update_graph(confidence_values):
    plt.clf()
    plt.plot(confidence_values)
    plt.xlabel('Window')
    plt.ylabel('Confidence')
    plt.title('Confidence Over Time')
    plt.pause(0.01)

match programMode:

    ########## Data Collection ##########
    case 0:
        inputListener.start()
        while True:
            if keyboard.is_pressed(killKey):
                inputListener.stop()
                os._exit(0)
            time.sleep(windowSize)
            inputListener.save_to_files(dataLabel)

    ########## Model Training ##########
    case 1:
        buttonLSTM = models.LSTMClassifier(inputSize=4,
                                           hiddenSize=32,
                                           classCount=2,
                                           layerCount=2,
                                           device=device).to(device)
        if captureMouse:
            moveLSTM = models.LSTMClassifier(inputSize=2,
                                             hiddenSize=32,
                                             classCount=2,
                                             layerCount=2,
                                             device=device).to(device)
        if captureController:
            stickLSTM = models.LSTMClassifier(inputSize=3,
                                              hiddenSize=32,
                                              classCount=2,
                                              layerCount=2,
                                              device=device).to(device)
            triggerLSTM = models.LSTMClassifier(inputSize=3,
                                                hiddenSize=32,
                                                classCount=2,
                                                layerCount=2,
                                                device=device).to(device)

        buttonTensor = torch.empty((0, 4))  # Button input (4 features)
        moveTensor = torch.empty((0, 2))    # Mouse movement (2 features)
        stickTensor = torch.empty((0, 3))   # Stick input (3 features)
        triggerTensor = torch.empty((0, 3)) # Trigger input (3 features)

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                dataFrame = pandas.read_csv(filePath)
                tensor = torch.tensor(dataFrame.values, dtype=dataType)
                
                if 'cheat' in fileName: # Decide on how to pass yTrain
                    if 'button' in fileName:
                        bTensor = torch.cat((buttonTensor, tensor), dim=0)
                    elif 'move' in fileName:
                        mTensor = torch.cat((moveTensor, tensor), dim=0)
                    elif 'stick' in fileName:
                        sTensor = torch.cat((stickTensor, tensor), dim=0)
                    elif 'trigger' in fileName:
                        tTensor = torch.cat((triggerTensor, tensor), dim=0)
                elif 'control' in fileName:
                    if 'button' in fileName:
                        bTensor = torch.cat((buttonTensor, tensor), dim=0)
                    elif 'move' in fileName:
                        mTensor = torch.cat((moveTensor, tensor), dim=0)
                    elif 'stick' in fileName:
                        sTensor = torch.cat((stickTensor, tensor), dim=0)
                    elif 'trigger' in fileName:
                        tTensor = torch.cat((triggerTensor, tensor), dim=0)

        if bTensor.size(0) > 0:
            buttonLSTM.train() # Add appropriate training params
            torch.save(buttonLSTM.state_dict(), 'models/button.pt')
        if mTensor.size(0) > 0:
            moveLSTM.train() # Add appropriate training params
            torch.save(moveLSTM.state_dict(), 'models/move.pt')
        if sTensor.size(0) > 0:
            stickLSTM.train() # Add appropriate training params
            torch.save(stickLSTM.state_dict(), 'models/stick.pt')
        if tTensor.size(0) > 0:
            triggerLSTM.train() # Add appropriate training params
            torch.save(triggerLSTM.state_dict(), 'models/trigger.pt')

    ########## Live Analysis ##########
    case 2: 
        inputListener.start()

        scaler = StandardScaler()
        buttonLSTM = torch.load('models/button.pt')
        if captureMouse:
            moveLSTM = torch.load('models/move.pt')
        if captureController:
            stickLSTM = torch.load('models/stick.pt')
            triggerLSTM = torch.load('models/trigger.pt')

        confidence_values = []
        if displayGraph:
            plt.ion()
            plt.figure()

        while True: # Or while game is running?
            if keyboard.is_pressed(killKey):
                inputListener.stop()
                if displayGraph: # You may want to save the graph even when displayGraph false
                    plt.savefig('confidence_graph.png')
                os._exit(0)
            time.sleep(windowSize)
            confidence = 1
            with torch.inference_mode():
                output = buttonLSTM(scaler.fit_transform(inputListener.buttonData))
                confidence *= torch.softmax(output, dim=1)[1] # Verify output format before doing this
                inputListener.buttonData = []
            if captureMouse:
                with torch.inference_mode():
                    output = moveLSTM(scaler.fit_transform(inputListener.moveData))
                    confidence *= torch.softmax(output, dim=1)[1] # Verify output format before doing this
                    inputListener.moveData = []
            if captureController:
                with torch.inference_mode():
                    output = stickLSTM(scaler.fit_transform(inputListener.stickData))
                    confidence *= torch.softmax(output, dim=1)[1] # Verify output format before doing this
                    inputListener.stickData = []
                    output = triggerLSTM(scaler.fit_transform(inputListener.triggerData))
                    confidence *= torch.softmax(output, dim=1)[1] # Verify output format before doing this
                    inputListener.triggerData = []
            confidence_values.append(confidence.item())
            print(confidence)
            if displayGraph:
                update_graph(confidence_values)

inputListener.stop()