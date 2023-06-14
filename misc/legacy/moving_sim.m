R = 0.55;
goal = [-92;
        24];
p_c = [-107.12; 
       67.90];
p_o = [-98.22;
       39.70;];
t = 0.05;
alpha = 1;
trust = [0.8];

data = [];
P = [p_c];

figure;
hold on

for i = 1: 2000
    
    H = eye(2) * 2;
    f = [0; 0];

    min_U = 0.2;
    max_U = 0.85;
    prop_gain = 0.85;
    U = (goal - p_c) * prop_gain;
        for j = 1: 2
            if abs(U(j)) < min_U && U(j) ~= 0 
                U(j) = (U(j)/abs(U(j))) * min_U;
            end
            if abs(U(j)) > max_U && U(j) ~= 0 
                U(j) = (U(j)/abs(U(j))) * max_U;
            end
        end
   
    diff = p_c - p_o;
    A = - 2 * diff';
    b = alpha * trust .* (sum(diff.^2)' - R^2) + 2 * diff' * U;
    lb = [-2; -2] - U;
    ub = [2; 2] - U;
    
    
    x = quadprog(H, f, A, b, [], [], lb, ub);
    p_c = p_c + (x + U) * t;
    P = [P p_c];
    colors = ["black", "blue", "yellow", "orange"];
    if mod(i, 50) == 3
        plot(P(1, :), P(2, :), "Color", "magenta");
        th = 0:pi/50:2*pi;
        legends = ["ego vehicle"];
        for k = 1: size(p_o, 2)
            plot(p_o(1, k), p_o(2, k), "o", "Color", colors(k));
            xunit = R * cos(th) + p_o(1, k);
            yunit = R * sin(th) + p_o(2, k);
            h = plot(xunit, yunit, "Color", "red");
            legends = [legends "trust:" + string(trust(k)) ""];
        end    
        plot(goal(1), goal(2), "x", "Color", "green");
        legends = [legends "Goal"];
        legend(legends);
        xlim([0, 7]);
        ylim([0, 7]);
        xlabel("X (m)");
        ylabel("Y (m)");
        title("Stationery obstacles with different trust");
        pause(0.1);
    end
end