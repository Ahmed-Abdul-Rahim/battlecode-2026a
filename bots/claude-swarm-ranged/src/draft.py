import cambc
from randomness import below

PREFS = [4, 5, 1, 10, 7]


def play():
    chosen = cambc.draft.current_draft(cambc.Team.ALLY)
    for pick in PREFS:
        if pick not in chosen:
            cambc.draft.pick(pick)
            return
    available = [pick for pick in range(11) if pick not in chosen]
    cambc.draft.pick(available[below(len(available))])
