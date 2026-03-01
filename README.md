# ToDo
  Code Implementation
 * Calibration_duration = 5 seconds
 * Disable countdown in calibration
 * check calibration buffer for size and if its getting data from true source in a true way.
 * Delete old SEM datas and folders and plots
 * Folders:
   * Rawdata: P1, P2, P3, P4
   * Rawdataplots: P1, P2, P3, P4
   * Prodata (Processed data): P1, P2, P3, P4
   * Prodataplot (processed data plots): P1, P2, P3
 * Data handler fix:
   * Datas will be written to shared memory when sEMG comes
   * sEMG data will be written taking mean average of them (assuming emg sends 100hz for both values) -> or find another way.
 * check expected hz in the program and data handler
 * new calibration method:
    *user will write cb and after instead of asking name the program will ask enter the values:
    user answer contains "folderResponsiveDirectory gesturename" they are seperated from space chrachters -> set folder and gesture name. Then 
    * when r pressed start recording.
    * when x pressed escape calibration loop

DO NOT FORGET THE RULES

# RULES: 
Please really read all the code. 
Do not make assumptions while answering. 

While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me.

Before implementing the whole system we may discuss the structure and do the necessary choices.

Create a list how this would be implemented to current system , which parts should be changed which parts should be added and where. 

Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.

When change of code only change what is relevant do not try to change anything that is not relevant with my aim. I need to know which parts you have made changes tell me because sometimes you change something inside a fucntion which I am not aware of.

You may only write which parts of the code I should change and where changes should be made to save time instead of writing the whole script again.

WHAT IS EXPECTED FROM THIS CODE. WHAT CAN IT DO. WHICH MISMATCHES HAVE YOU SPOTTED AS A LIST


# system overview

main.py the file user will run also where user will give commands frequently
it initializes processes. getdata process and main.py is always running
sckit-learn would do classificaiton,training etc. when triggered
I need to have a class for sckitlearn which has methods like classify,train then call them inside main .py
Also I need to save data (calibration) for gestures . Since I will be doing dynamic and static gestures we need a time series or some time related things in order to use.
Also we need to get data not using connection id we need to get data with device name

classification and commanding should run simulteneusly so I can start-stop classsification. I would prefer multiprocessing.

we will have 2 models which they will need seperate classes restmodel and classifymodel class:
model1(the rest model) in rest_model1.py It can accurately recognize if a gesture is rest or not.

----------------- already done left to gain basic understanding of the system --------------------------

