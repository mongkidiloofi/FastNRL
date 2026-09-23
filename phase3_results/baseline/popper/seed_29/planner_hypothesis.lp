state_at(B,T+1):-time(T),adjacent(up,B,C),state_at(C,T),action(up,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(up,C,B),state_at(B,T),action(up,T),wall(C).
state_at(B,T+1):-time(T),adjacent(left,B,C),state_at(C,T),action(left,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(down,C,B),state_at(B,T),action(down,T),wall(C).
state_at(B,T+1):-time(T),adjacent(right,C,B),state_at(B,T),action(right,T),wall(C).
state_at(B,T+1):-time(T),adjacent(right,B,C),state_at(C,T),action(right,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(down,B,C),state_at(C,T),action(down,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(left,C,B),state_at(B,T),action(left,T),wall(C).
