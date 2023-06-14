R = 0.55;
goal = [3.30;
        6.50];
p_c = [3; 
       0];
% p_o = [2.75 4.20 3.50;
%        4.00 4.20 5.80;];
p_o = [2.20 4.20;
       4.00 4.00;];
% u_o = [0.06 0.07 -0.03;
%        0.00 0.00 0.00];
u_o = [0.06 -0.06;
       0.00 0.00;];
t = 0.02;
gamma = 0.2;
d = 3.5;
% trust = [0.1; 0.9; 0.5];
trust = [0.23; 0.91];
alpha = zeros(size(trust));
for j = 1: size(trust)
    alpha(j) = 1/(1/gamma + d*(exp(-1*trust(j)) - 1)); 
end

data = [];
P = [p_c];

f = figure;
f.Position = [200 130 1000 870];
hold on

h = zeros(1, size(p_o, 2));
d = zeros(1, size(p_o, 2));
e = 1;
first = 0;

legends = ["ego vehicle"];


for i = 1: 2500
    
    H = eye(2) * 2;
    f = [0; 0];

    min_U = 0.2;
    max_U = 0.65;
    prop_gain = 0.65;
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
    b = alpha .* (sum(diff.^2)' - R^2) + 2 * diff' * U;
    lb = [-2; -2] - U;
    ub = [2; 2] - U;
    
    
    x = quadprog(H, f, A, b, [], [], lb, ub);
    
    colors = ["black", "blue", "yellow", "orange"];
    if mod(i, 50) == 3
        if first ~= 0
            delete(h); delete(d); delete(ex);
            h = zeros(1, size(p_o, 2));
            d = zeros(1, size(p_o, 2));
        end        

        if first == 0
            plot(goal(1), goal(2), "x", "Color", "green", "LineWidth", 2.5, "MarkerSize", 15);
            legends = ["Goal" legends ];
        else
            legends = [legends ""];
        end    
        ex = plot(P(1, :), P(2, :), "Color", "magenta", "LineWidth", 2.5);
            th = 0:pi/50:2*pi;
%         legends = ["ego vehicle"];
        for k = 1: size(p_o, 2)
            
            xunit = R * cos(th) + p_o(1, k);
            yunit = R * sin(th) + p_o(2, k);
            d(k) = plot(p_o(1, k), p_o(2, k), "o", "Color", colors(k), "LineWidth", 3, "MarkerSize", 8);

            h(k) = plot(xunit, yunit, "Color", "red", "LineWidth", 2.5);
            if first == 0
            legends = [legends "trust:" + string(trust(k)) ""];
            else
            legends = [legends "" ""];
            end
        end    
        

        first = 1;
        legend(legends, "FontSize", 20);
        xlim([0, 7]);
        ylim([0, 7]);
        xlabel("X (m)");
        ylabel("Y (m)");
        title("Moving obstacles with different trust");
        pause(0.1);
    end

    p_c = p_c + (x + U) * t;
    P = [P p_c];
    p_o = p_o + u_o * t;
end