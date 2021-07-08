# basic PySimpleGUI
# Example 01 - The One-Shot Window
import PySimpleGUI as sg                        # Part 1 - The import

# Define the window's contents
layout = [[sg.Text("What's your name?")],     # Part 2 - The Layout
          [sg.Input()],
          [sg.Button('Ok')]]

# Create the window
window = sg.Window('Window Title', layout)      # Part 3 - Window Defintion

# Display and interact with the Window
# Part 4 - Event loop or Window.read call
event, values = window.read()

# Do something with the information gathered
print('Hello', values[0], "! Thanks for trying PySimpleGUI")

# Finish up by removing from the screen
window.close()
