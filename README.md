# TpM-Agro: IoT and Machine Learning Applied to Vertical Farms

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- [![GitHub last commit](https://img.shields.io/github/last-commit/WissTek-IoT/TpM-Agro)](#) -->

![Aeroponics Greenhouse](https://drive.google.com/thumbnail?id=1BcZDbsrD4igT2k8ysZTfdPOcN1ZeqeU2&sz=w1000)

## 🗒️Table of Contents
- [About](#-about)
    - [Functionalities](#️-functionalities)
    - [Folder Structure](#-folder-structure)
    - [Results](#-results)
    - [Personal Outcomes](#-personal-outcomes)
    - [Gallery](#️-gallery)
- [License](#️-license)
- [Feedback and Contributions](#-feedback-and-contributions)
- [Contact](#️-contact)
 
## 🔍️ About

> This was an undergraduate research project conducted at the Faculty of Electrical and Computer Engineering (FEEC) of the Universidade Estadual de Campinas - UNICAMP from 09/2023 to 08/2024.

Vertical farms can reduce environmental impact caused by conventional agriculture while also increasing productivity. For example, [aeroponics systems can grow plants without soil and use up to 98% less water than traditional methods](https://attra.ncat.org/publication/vertical-farming/). 

[![Aeroponics System](https://ponicslife.com/wp-content/uploads/2022/03/Aeroponic-System-1024x619.jpg)](https://ponicslife.com/what-is-aeroponics-everything-you-need-to-know/)

However, the need for a controlled environment makes urban farming solutions expensive and limited to specialists. But... *what if we could develop a low-cost greenhouse and train a machine learning model to control it as an expert would?* **That's exactly what this project aimed to do**. 

### 🚀 Project Description
A comprehensive functional diagram can be seen below.

![Functional Diagram](https://drive.google.com/thumbnail?id=1FYeUbSg2Q41wQyWoDVmYaU6PWCLK3PzB&sz=w1000)

The **process device** ([a Seeeduino v3.0 board](https://wiki.seeedstudio.com/Seeeduino_v3.0/)) measures temperature, humidity and light intensity. Those values are assembled on a packet (**process data**) and sent to the **prediction device** ([a Raspberry Pi 3B](https://www.raspberrypi.com/products/raspberry-pi-3-model-b/)). The prediction device uses the process data to generate additional features, such as the rate of change of each variable or elapsed seconds since the process started. Those extra features are called the **abstraction data**. Process data and abstraction data are then merged to form the **system data**, which is used to feed three independent Deep Neural Networks (DNNs):

- A **binary-classification model** to predict lamp activations
- A **regression model** to predict the time interval between nutrition cycles
- A **regression model** to predict the duration of a nutrition cycle

These models operate in real-time, generating predictions every second (configurable by the user). Those predictions are stored in a packet (**control signal**) and sent to the process device, which, according to those signals, controls the grow light and nutrient pump.

The user can interact with the process device by controlling if the system will run manually (i.e, a human specialist controlling the actuators during the learning stage) or automatically (by the DNNs) using physical buttons.

![Buttons](https://drive.google.com/thumbnail?id=12BgW8t7CJ-n2oNIYW6h33sREP9vpUfas&sz=w1000)
*From left to right: enable pump, turn grow light on/off, and switch from manual to automatic control (blue LED on means automatic control).*

It is also possible to interact with the prediction device through a command screen, where you can both set system parameters or visualize the predicted signals.
![Commands](https://drive.google.com/thumbnail?id=1up52P9AZsAGhNc6KXAVSlF5KoMN3nhlU&sz=w1000)

For five days, the system was controlled by an aeroponics specialist, with data being stored each second. This resulted in more than 226 thousand samples, split into 70% for training, 15% for validation, and 15% for testing the chosen models. All models were trained on a personal computer. After the models were chosen, a tflite version was generated so it could be used on the abstraction device to perform inferences based on new data. All these features are included on a single *abstraction.py* file.

### 📂 Folder Structure
Folder names were given based on the [Three-Phase Methodology for IoT Project Development](https://www.sciencedirect.com/science/article/abs/pii/S2542660522001056#:~:text=It%20is%20a%20generic%2C%20agile,Business%2C%20Requirements%2C%20and%20Implementation.).
```
.
├── L1_Device                  # Process Device
│   ├── ...
│   ├── include                
│   │   ├── pinout.h           # Pinout definitions for the process device
│   │   ├── utils.h            # Common-used libraries
│   │   └── ...
│   ├── lib                    # Internal libraries organized by functionality
│   │   ├── light              # Grow light activation control
│   │   ├── mode               # Manual/Automatic mode management
│   │   ├── MoT                # Communication protocol library
│   │   ├── pump               # Pump control
│   │   ├── WTK_TH             # Integrated temperature and humidity sensor
│   │   └── ...
│   ├── src                    # Main source code for the device
│   └── ...

├── L3_Border                  # Interface between the process and abstraction device
│   └── border.py              # Reads and stores process data

├── L4_Storage                 # Stores system data
│   ├── abstraction_data.txt   
│   ├── application_data.txt   
│   ├── bkp_app_data.txt       # Backup of application data
│   ├── commands.txt           # Commands terminal
│   ├── light_model.keras      # DNN model for light control
│   ├── light_model.tflite     # Same as above, in tflite format
│   ├── pump_activating_model.keras  # DNN model to predict the duration of a nutrition cycle
│   ├── pump_activating_model.tflite # Same as above, in tflite format
│   ├── pump_waiting_model.keras     # DNN model to predict the interval between nutrition cycles
│   ├── pump_waiting_model.tflite    # Same as above, in tflite format
│   ├── prediction_queue.txt    # New system data to generate control output
│   ├── testing_data.txt        # Data for testing the models
│   ├── training_data.txt       # Data for training the models
│   └── validation_data.txt     # Data for model validation

├── L5_Abstraction             # Data abstraction and processing logic
│   └── abstraction.py         # Compute abstraction data, train and run all three models

└── L6_Exibition               # Data display 
    └── exibition.py           # Not implemented
```

### ✨ Results
The metrics for the three DNNs are as follows:

- Global accuracy of 98,8% for the light prediction DNN
- Mean absolute error of 60 seconds (3,36%) for the interval between prediction cycles DNN
- Mean absolute error of 3 seconds (10%) for the nutrition cycle duration DNN

![PW Model Graph](https://drive.google.com/thumbnail?id=15zmJsnq9t7DMePo0Mp9WDieLQ3eVBSwX&sz=w1000)
*Predicted interval between nutrition cycles in minutes versus elapsed time in hours. The black curve is the reference. The green curve is the model output. The orange curve is the grow light state.*

While running in automatic control mode, the system prints relevant information to the user.

![Rpi output](https://drive.google.com/thumbnail?id=1YxF9vKe9WELR8zCCSIUOQlNMvze6lDzj&sz=w1000)

The system was put in automatic control for five days, resulting in outstanding crop growth.

![Crop growth](https://drive.google.com/thumbnail?id=1M5acUA74OQl7BGY1pJ182D_xlM0Gz-Kx&sz=w1000)
*T stands for training stage (manual control). C stands for automatic control stage.*

### 📝 Personal Outcomes

Developing a multidisciplinary project is always challenging but comes with valuable rewards in the end. On this journey, I enhanced my programming skills in Arduino by implementing and debugging communication protocols for IoT device interaction, as well as creating code libraries for the sensors and actuators used in the project. Assembling the electronics and designing basic circuits for the manual control stage added an enjoyable, hands-on component.

My most significant achievements, however, were in Machine Learning. TpM-Agro was my first project involving ML, and I can honestly say it was an incredible experience. Tackling a practical problem turned out to be an engaging and enjoyable way to dive into the field, as it encouraged me to explore various techniques to optimize my models. At first, I thought a single multi-class DNN could handle everything, but as you can see, that wasn’t the case. The goal of achieving satisfactory performance metrics led me to dive deep into different prediction models, feature engineering, overfitting issues, how to run effective diagnostics, balancing bias and variance, and managing the trade-offs between precision and recall. Although I still have much to learn, this challenging journey taught me a lot about ML model design.

There were also unexpected outcomes along the way—like learning the optimal light cycle for lettuce growth, assembling the aeroponics setup, and troubleshooting a nutrient pump that occasionally decided not to work.

Overall, this project was a fantastic experience, and I’m grateful for the learning and growth it brought me.

### 🖼️ Gallery
![Growth comparison](https://drive.google.com/thumbnail?id=1Bjbf9ezuKbFyld-OtD1tMFfOAcWRrlPb&sz=w1000)
*Plant growth from T01 (left) to C05 (right)*

![Microcontrollers](https://drive.google.com/thumbnail?id=1EWFaC8q7ctP4wdO0ZReYIbw4pJjScL00&sz=w1000)
*Process Device (Seeeduino v3.0) and abstraction device (Raspberry Pi 3B)*

![Early plants](https://drive.google.com/thumbnail?id=1UmaRTyfOV1cwJMQExNrOlGGRbHSRF2PB&sz=w1000)
*Plants in T01*

![Sensors](https://drive.google.com/thumbnail?id=1AnsLlIjtvzdLWZmtSIB3BNHeaYXN1LXj&sz=w1000)
*Light Sensor (right) and temperature and humidity sensor (left)*

![Box](https://drive.google.com/thumbnail?id=1jevshx4NXFtZonPOK-lBejVnMqjJXamG&sz=w1000)
*Nutrient pump*

## ⚖️ License
This project is licensed under the Apache License 2.0. You are free to use, modify, and distribute this code, provided that proper attribution is given, and the license terms are included in any distributions.

## 🤝 Feedback and Contributions
Since TpM-Agro was developed as a 1-year research project, I do not plan to make further improvements to the system or code. However, you’re welcome to explore it, experiment with it, and share any suggestions or improvements!

## 💬 Contact
[
    <img
        src="https://images.weserv.nl/?url=https://github.com/vdrad.png&fit=cover&mask=circle&maxage=7d" 
        width=10%
        title="GitHub Profile"
        alt="vdrad"
    />
](https://github.com/vdrad)

For questions, suggestions, collaboration opportunities, or any other topic, please feel free to reach out:

- GitHub:   [vdrad](https://github.com/vdrad)
- Email:    victor.drad.g@gmail.com
- LinkedIn: [Victor Gomes](https://www.linkedin.com/in/victor-g-582b5911b/)

