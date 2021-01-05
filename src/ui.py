import PySimpleGUI as sg

# Constants
BOARDLEN = 648


control_col = sg.Column([[sg.Text('settings:')],
                         [],
                         [sg.Text('current move:')],
                         [],
                         ])

board_col = sg.Column([[sg.Text('board:')],
                       [sg.Graph(canvas_size=(BOARDLEN, BOARDLEN), graph_bottom_left=(
                           0, 0), graph_top_right=(BOARDLEN, BOARDLEN), background_color='lightgrey', key='board')]
                       ])
history_col = sg.Column([[sg.Text('history:')],
                         [sg.Output(size=(15, 40), key='-HISTORY-')]])

layout = [
    [control_col, board_col,
     history_col],
    # sg.Row([sg.Text('Console'), sg.Print(
    #    'Re-routing the stdout', do_not_reroute_stdout=True)]),
    [sg.Button('Play'), sg.Button('Exit')]
]

window = sg.Window('twixtbot-ui', layout)

while True:             # Event Loop
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
window.close()
