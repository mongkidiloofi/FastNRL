transition(A,B):- state_before(A,C),step_left(A,B,C),open(A,B).
transition(A,B):- state_before(A,B),step_left(A,C,B),wall(A,C).
transition(A,B):- state_before(A,B),step_right(A,C,B),wall(A,C).
transition(A,B):- state_before(A,B),step_up(A,C,B),wall(A,C).
transition(A,B):- state_before(A,C),step_right(A,B,C),open(A,B).
transition(A,B):- state_before(A,C),step_down(A,B,C),open(A,B).
transition(A,B):- state_before(A,C),step_up(A,B,C),open(A,B).
transition(A,B):- state_before(A,B),step_down(A,C,B),wall(A,C).
