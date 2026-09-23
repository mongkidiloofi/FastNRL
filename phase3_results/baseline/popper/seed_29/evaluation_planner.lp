1{action(down, T); action(up, T); action(right, T); action(left, T)}1 :- time(T), not finished(T).
#show state_at/2.
 #show action/2.
%CCC
time(0..250).
%DDD
cell((0..6,0..5)).
#minimize{1, X, T: action(X,T)}.
adjacent(right, (X+1,Y),(X,Y))   :- cell((X,Y)), cell((X+1,Y)).
adjacent(left,(X,Y),  (X+1,Y)) :- cell((X,Y)), cell((X+1,Y)).
adjacent(down, (X,Y+1),(X,Y))   :- cell((X,Y)), cell((X,Y+1)).
adjacent(up,   (X,Y),  (X,Y+1)) :- cell((X,Y)), cell((X,Y+1)).
%START
state_at(B,T+1):-time(T),adjacent(up,B,C),state_at(C,T),action(up,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(up,C,B),state_at(B,T),action(up,T),wall(C).
state_at(B,T+1):-time(T),adjacent(left,B,C),state_at(C,T),action(left,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(down,C,B),state_at(B,T),action(down,T),wall(C).
state_at(B,T+1):-time(T),adjacent(right,C,B),state_at(B,T),action(right,T),wall(C).
state_at(B,T+1):-time(T),adjacent(right,B,C),state_at(C,T),action(right,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(down,B,C),state_at(C,T),action(down,T),not wall(B).
state_at(B,T+1):-time(T),adjacent(left,C,B),state_at(B,T),action(left,T),wall(C).
%END
%AAA
state_at((1, 4), 1).
%BBB
finished(T):- goal(T2), time(T), T >= T2.
 goal(T):- state_at((5, 1), T), not finished(T-1).
goalMet:- goal(T).
:- not goalMet.
wall((0, 0)).
wall((0, 1)).
wall((0, 2)).
wall((0, 3)).
wall((0, 4)).
wall((0, 5)).
wall((1, 0)).
wall((1, 3)).
wall((1, 5)).
wall((2, 0)).
wall((2, 1)).
wall((2, 2)).
wall((2, 3)).
wall((2, 5)).
wall((3, 0)).
wall((3, 3)).
wall((3, 5)).
wall((4, 0)).
wall((4, 2)).
wall((4, 3)).
wall((4, 5)).
wall((5, 0)).
wall((5, 5)).
wall((6, 0)).
wall((6, 1)).
wall((6, 2)).
wall((6, 3)).
wall((6, 4)).
wall((6, 5)).
