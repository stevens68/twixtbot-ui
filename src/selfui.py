
import argparse
#import numpy

import twixt
import ui2 as ui
import graphics as gr


parser = argparse.ArgumentParser(description='One twixt position.')
parser.add_argument('--moves', '-m', type=str)
parser.add_argument('--check_draw', '-D', action='store_true')
parser.add_argument('--think_report', '-T', action='store_true')
parser.add_argument('--show_game_state', '-G', action='store_true')
args = parser.parse_args()


game = twixt.Game()
trendline = []
forecast = 0.0
expLine = ""


if args.moves:
    for m in args.moves.split(','):
        game.play(m)
        trendline.append(0)


w = ui.TwixtBoardWindow()
w.set_game(game)
gr.update()
c = ui.TwixtControlWindow()

TRIALS = 100
thinkerargs = "nnmplayer:trials=" + \
    str(TRIALS) + ",model=model\pb,location=C:\temp\loc1"
thinker = twixt.get_thinker(thinkerargs, None)


while True:

    w.win.getMouse()
    c.reset(game)
    gr.update()

    tup = thinker.pick_move(game, c)

    # print "REPORT:" , rep, len(rep)
    report = thinker.report.split(" :pv=")

    if len(report) == 2:
        forecast = float(report[0])
        forecast = -forecast if len(game.history) % 2 == 1 else forecast
        expLine = report[1]
    else:
        expLine = ""

    trendline.append(forecast)

    if type(tup) == tuple:
        m, n = tup
    else:
        m = tup
    print(m)
    if m != "resign":
        game.play(m)
    else:
        break

    # update board window
    c.progressBar['value'] = 100.0
    c.update(game, forecast, trendline, expLine)
    w.create_move_objects(game, len(game.history) - 1)
    gr.update()

    if game.is_winning(game.BLACK) or game.is_winning(game.WHITE):
        break

    if not game.can_win(game.WHITE) and not game.can_win(game.BLACK):
        break

w.win.getMouse()
