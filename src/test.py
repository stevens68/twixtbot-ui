import PySimpleGUI as sg

tab1_layout = [[sg.T('This is inside tab 1')]]

tab2_layout = [[sg.T('This is inside tab 2')],
               [sg.In(key='in')]]

layout = [[sg.TabGroup([[sg.Tab('Tab 1', tab1_layout, tooltip='tip'), sg.Tab('Tab 2', tab2_layout)]], tooltip='TIP2')],
          [sg.Button('Read')]]

window = sg.Window('My window with tabs', layout, default_element_size=(12, 1))

while True:
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED:           # always,  always give a way out!
        break
