# -*- coding: utf-8 -*-
"""
Created on 2/3/21
@author: MONTGM11

Echo_Segmenter_UI_Full.py
User interface for automated echocardiography tool. Combines b-mode and m-mode tools
"""

# Import libraries
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox as msg
import os

# Create GUI
window = tk.Tk()
window.title('Echocardiography Segmenter')
window.geometry('900x200')
# Title
appTitle = tk.Label(window,text='Echocardiography Segmenter',font=('calibri',20,'bold'),fg='forest green')
appTitle.grid(column=3,row=0)


# ----------------------- Formatting ----------------------- 
# Create offset at top and left
offsetBlank = tk.Label(text='',width = 4,height = 2)
offsetBlank.grid(column=0,row=0)
# Create space between rows
blankRow2 = tk.Label(text='',width = 1, height = 1)
blankRow2.grid(column=0,row=2)
blankRow2 = tk.Label(text='',width = 1, height = 1)
blankRow2.grid(column=0,row=4)
blankRow2 = tk.Label(text='',width = 1, height = 1)
blankRow2.grid(column=0,row=6)
blankRow2 = tk.Label(text='',width = 1, height = 1)
blankRow2.grid(column=0,row=8)
blankRow2 = tk.Label(text='',width = 1, height = 1)
blankRow2.grid(column=0,row=12)

# --------------------- Button to select study directory ---------------------

# Define callback
def button1clicked():
    # Have user select input file
    studyDir.set(filedialog.askdirectory())
    studyDirNameDisp.configure(text=studyDir.get(),justify='left')
    if len(studyDir.get()) > 0:
        print(studyDir.get() + ' selected')

    else:
        print('No file found')        

# Create button, label, and text displaying result
button1Label = tk.Label(window,text='Input Directory:',justify='right',width=20)
button1Label.grid(column=1,row=1)
studyDir = tk.StringVar()
studyDir.set('None Selected')
studyDirNameDisp = tk.Label(window,textvariable=studyDir,font=('arial',8),justify='center',bg='white',width=90)
studyDirNameDisp.grid(column=2,row=1,columnspan=3)
button1 = tk.Button(window,text='Select',command=button1clicked,justify='right',width=10)
button1.grid(column=5,row=1)

# ------------------------ Verbose Check Box ------------------------------
# M-Mode
verbose = tk.BooleanVar()
verbose.set(True)
chk1 = ttk.Checkbutton(window,text='Save QC Results',var=verbose)
chk1.grid(column=3,row=3)


# ----------------------- Run button ----------------------------
def runButtonClicked():
    # This runs the main program
    print('Running analysis...')
    #os.chdir("D:/Echo_Segmenter_Full/")
    os.chdir('X:/Mary Kate/ECHO/Echo_Analysis/Echo_Segmenter_Test/')
    from preprocessing_modes import sort_modes, write_command
    
    # Compose command string to feed into os system
    comm1 = 'activate tf-latest'
    
    # If verbose, add flag to commands
    if verbose.get():
        verboseString = '--verbose '
    else:
        verboseString = ''
    
    # Parse input directory
    runMode = sort_modes(studyDir.get())
    
    # If no data in folder, alert user and exit
    if runMode == 'None':
        msg.showinfo(message='No Data Found')
        return
    
    # If M-Mode data present, add command to run M-Mode analysis. 
    comm2 = write_command(studyDir.get(),runMode,'M-Mode',verboseString)
    
    # If B-Mode data present, add command to run B-Mode analysis. 
    comm3 = write_command(studyDir.get(),runMode,'B-Mode',verboseString)
        
    # Combine into 1 command string
    commandString = comm1 + comm2 + comm3
    
    # Run the command to start the main program
    a = os.system(commandString)
    # Update user
    if a==0:
        cmpltMsg = 'Analysis Complete!'
    else:
        cmpltMsg = 'There was an issue with this run. CMD error code: ' + str(a)
    print(cmpltMsg)
    msg.showinfo(message=cmpltMsg)
    
        
runButton = tk.Button(window,text='Run',font=('Arial',18),command=runButtonClicked)
runButton.grid(column=3,row=5)

# Main
window.mainloop()