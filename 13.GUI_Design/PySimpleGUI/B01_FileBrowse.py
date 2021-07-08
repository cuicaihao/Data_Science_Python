import PySimpleGUI as sg

sg.theme('Dark Blue 3')  # please make your creations colorful

layout = [[sg.Text('Filename'), sg.Input(key='-sourcefile-'), sg.FileBrowse()],
          [sg.OK('OK'), sg.Button('Exit')]]

window = sg.Window('Get filename example', layout)

while True:
    event, values = window.read()
    if event in ('Exit', 'Quit', None):
        break
    source_file = values['-sourcefile-']
    if event is 'OK':
        sg.popup("Source file is:", source_file + str(type(source_file)))


window.close()
