function U = braindeadController(X, track, verbose)

    Kp = 1;
    pos_diff = track.cline - [X(1); X(3)];
    [closest_point, ind] = min(sqrt(sum(pos_diff.^2, 1)));
    
    heading = atan2(track.cline(2,ind+1)-track.cline(2,ind), track.cline(1,ind+1)-track.cline(1,ind));
    e_psi = heading - X(5);

    U = zeros(10,2);
    U(:,1) = e_psi * Kp;
    U(:,2) = 100 - X(2);

    if verbose
        disp(ind)
        sprintf("e_psi = %d", e_psi)
    end
end
