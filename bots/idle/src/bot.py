"""Idle opponent for harness/speedrun.py: champions never act.

Against it a bot farms uncontested, so the speedrun measures pure farming
speed (rounds to the winning gem count), independent of any opponent.
"""


class Bot:
    def play_turn(self):
        return
