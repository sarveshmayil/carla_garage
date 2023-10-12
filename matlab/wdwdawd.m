clc; clear; close all

load("TestTrack.mat");

X = [287,5,-176,0,2,0];

figure
hold on
plot(TestTrack.bl(1,:), TestTrack.bl(2,:))
plot(TestTrack.br(1,:), TestTrack.br(2,:))
trajectory = animatedline(X(1), X(3));

I = 0.0;
prev_error = 0.0;

verbose = true;
while true
    [U,I,prev_error] = braindeadController2(X, I, prev_error, TestTrack, verbose);
    % U = braindeadController(X, TestTrack, verbose);
    [X, T] = forwardIntegrateControlInput(U, X);
    X = X(end,:);
    addpoints(trajectory, X(1), X(3))
    disp(X)
    drawnow
end
