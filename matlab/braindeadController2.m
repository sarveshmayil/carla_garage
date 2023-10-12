function [U,I,error] = braindeadController2(X, I, prev_error, track, verbose)

    Kp = 0.6;
    Ki = 0.05;
    Kd = 0.03;
    dt = 0.01;

    pos_diff = track.cline - [X(1); X(3)];
    [closest_point, ind] = min(sqrt(sum(pos_diff.^2, 1)));
    
    heading = atan2(track.cline(2,ind+1)-track.cline(2,ind), track.cline(1,ind+1)-track.cline(1,ind));
    e_psi = heading - X(5);
    error = e_psi;

    P = Kp * e_psi;
    I = I + Ki * e_psi;
    D = Kd * (e_psi - prev_error) / dt;

    U = zeros(10,2);
    U(:,1) = P + I + D;
    U(:,2) = 100 - X(2);

    if verbose
        disp(ind)
        sprintf("e_psi = %d, P = %d, I = %d, D = %d", e_psi, P, I, D)
    end
end
